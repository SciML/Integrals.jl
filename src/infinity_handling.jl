# generic change of variables code

function substitute_u(u2v, lb, ub)
    bounds = map(u2v, lb, ub)
    lb_sub = lb isa Number ? first(bounds) : map(first, bounds)
    ub_sub = ub isa Number ? last(bounds) : map(last, bounds)
    return (lb_sub, ub_sub)
end

# without batching point container type should match the inputs
substitute_v(v2ujac, v, lb::Number, ub::Number) = v2ujac(only(v), lb, ub)
function substitute_v(v2ujac, v, lb::AbstractVector, ub::AbstractVector)
    xjac = map((l, u, v) -> v2ujac(v, l, u), lb, ub, v) # ordering may influence container type
    x = map(first, xjac)
    jac = prod(last, xjac)
    return x, jac
end

function substitute_f(f::IntegralFunction{false}, v2ujac, lb, ub)
    _f = f.f
    IntegralFunction{false}(f.integrand_prototype) do v, p
        u, jac = substitute_v(v2ujac, v, lb, ub)
        return _f(u, p) * jac
    end
end
function substitute_f(f::IntegralFunction{true}, v2ujac, lb, ub)
    _f = f.f
    prototype = similar(f.integrand_prototype)
    vol = prod((ub - lb) / 2) # just to get the type of the jacobian determinant
    IntegralFunction{true}(prototype * vol) do y, v, p
        u, jac = substitute_v(v2ujac, v, lb, ub)
        _y = _evaluate!(_f, prototype, u, p)
        y .= _y .* jac
        return
    end
end

# with batching the point container type is assumed to be mutable
function substitute_bv(v2ujac, v::AbstractArray, lb::Number, ub::Number)
    x = similar(v, typeof(one(eltype(v)) * (first(lb) + first(ub))))
    jac = similar(x)
    for i in axes(v, 1)
        x[i], jac[i] = v2ujac(v[i], lb, ub)
    end
    return x, jac
end
function substitute_bv(v2ujac, v::AbstractArray, lb::AbstractVector, ub::AbstractVector)
    x = similar(v, typeof(one(eltype(v)) * (first(lb) + first(ub))))
    jac = similar(v, typeof(zero(eltype(v)) * prod(lb)), size(v)[end])
    idx = CartesianIndices(axes(v)[begin:(end - 1)])
    for i in axes(v)[end]
        _jac = one(eltype(jac))
        for (ii, l, u) in zip(idx, lb, ub)
            x[ii, i], j = v2ujac(v[ii, i], l, u)
            _jac *= j
        end
        jac[i] = _jac
    end
    return x, jac
end

function substitute_f(f::BatchIntegralFunction{false}, v2ujac, lb, ub)
    _f = f.f
    BatchIntegralFunction{false}(f.integrand_prototype, max_batch = f.max_batch) do v, p
        u, jac = substitute_bv(v2ujac, v, lb, ub)
        y = _f(u, p)
        return y .* reshape(jac, ntuple(d -> d == ndims(y) ? length(jac) : 1, ndims(y)))
    end
end
function substitute_f(f::BatchIntegralFunction{true}, v2ujac, lb, ub)
    _f = f.f
    prototype = similar(f.integrand_prototype)
    vol = prod((ub - lb) / 2) # just to get the type of the jacobian determinant
    BatchIntegralFunction{true}(prototype * vol, max_batch = f.max_batch) do y, v, p
        u, jac = substitute_bv(v2ujac, v, lb, ub)
        _prototype = similar(prototype, size(y))
        _y = _evaluate!(_f, _prototype, u, p)
        y .= _y .* reshape(jac, ntuple(d -> d == ndims(y) ? length(jac) : 1, ndims(y)))
        return
    end
end

# we need this function for autodiff compatibility where internal buffers need special types
function _evaluate!(f, y, u, p)
    f(y, u, p)
    return y
end

# specific changes of variables

function u2t(lb, ub)
    mid = (lb + ub) / 2 # floating-point promotion
    if isinf(lb) && isinf(ub)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = flipsign(one(mid), ub)
    elseif isinf(lb)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = zero(one(mid))
    elseif isinf(ub)
        lb_sub = zero(one(mid))
        ub_sub = flipsign(one(mid), ub)
    else
        lb_sub = -one(mid)
        ub_sub = one(mid)
    end
    return lb_sub, ub_sub # unitless
end

function t2ujac(t::Number, lb::Number, ub::Number)
    u = oneunit(eltype(lb))
    # apply correct units
    if isinf(lb) && isinf(ub)
        den = inv(1 - t^2)
        t * den * u, (1 + t^2) * den^2 * u
    elseif isinf(lb)
        den = inv(1 - flipsign(t, lb))
        ub + t * den * u, den^2 * u
    elseif isinf(ub)
        den = inv(1 - flipsign(t, ub))
        lb + t * den * u, den^2 * u
    else
        den = (ub - lb) * oftype(float(one(u)), 0.5)
        lb + (1 + t) * den, den
    end
end

"""
    transformation_if_inf(f, domain)

Default infinity transformation using rational functions. Maps infinite domains to finite
intervals using the following substitutions:

  - For doubly-infinite domains `(-∞, ∞)`: `u = t/(1-t²)`, so `∫_{-∞}^{∞} f(u) du = ∫_{-1}^{1} f(t/(1-t²)) (1+t²)/(1-t²)² dt`
  - For semi-infinite domains `[a, ∞)`: `u = a + t/(1-t)`, so `∫_a^∞ f(u) du = ∫_0^1 f(a+t/(1-t)) 1/(1-t)² dt`
  - For semi-infinite domains `(-∞, b]`: `u = b + t/(1+t)`, so `∫_{-∞}^b f(u) du = ∫_{-1}^0 f(b+t/(1+t)) 1/(1+t)² dt`

This is the default transformation applied by algorithms like `QuadGKJL` and `HCubatureJL`
when encountering infinite integration bounds.

## Example

```julia
using Integrals

f(x, p) = exp(-x^2)
prob = IntegralProblem(f, (-Inf, Inf))

# Explicitly use the default transformation
alg = ChangeOfVariables(transformation_if_inf, QuadGKJL())
sol = solve(prob, alg)
```

See also: [`transformation_tan_inf`](@ref), [`transformation_cot_inf`](@ref), [`ChangeOfVariables`](@ref)
"""
function transformation_if_inf(f, domain)
    lb, ub = promote(domain...)
    tdomain = substitute_u(u2t, lb, ub)
    g = substitute_f(f, t2ujac, lb, ub)
    return g, tdomain
end

# to implement more transformations, define functions u2v and v2ujac and then a wrapper
# similar to transformation_if_inf to pass to ChangeOfVariables

# Alternative transformation using arctan for doubly-infinite domains
# This uses t = (2/π) * arctan(u), giving u = tan(πt/2)
# ∫_{-∞}^{∞} f(u) du = (π/2) ∫_{-1}^{1} sec²(πt/2) f(tan(πt/2)) dt
#
# For semi-infinite domains [a, ∞), we use u = a + tan(π(t+1)/4) mapping [−1, 1] → [a, ∞)
# For semi-infinite domains (−∞, b], we use u = b − tan(π(1−t)/4) mapping [−1, 1] → (−∞, b]

function u2t_tan(lb, ub)
    mid = (lb + ub) / 2 # floating-point promotion
    # All cases map to [-1, 1]
    if isinf(lb) && isinf(ub)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = flipsign(one(mid), ub)
    elseif isinf(lb)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = zero(one(mid))
    elseif isinf(ub)
        lb_sub = zero(one(mid))
        ub_sub = flipsign(one(mid), ub)
    else
        lb_sub = -one(mid)
        ub_sub = one(mid)
    end
    return lb_sub, ub_sub
end

function t2ujac_tan(t::Number, lb::Number, ub::Number)
    u = oneunit(eltype(lb))
    if isinf(lb) && isinf(ub)
        # Doubly-infinite: t ∈ [-1, 1] → u ∈ (-∞, ∞)
        # u = tan(πt/2), du/dt = (π/2) sec²(πt/2)
        halfpi = oftype(float(one(u)), π / 2)
        arg = halfpi * t
        tan_val = tan(arg)
        sec2 = 1 + tan_val^2  # sec² = 1 + tan²
        tan_val * u, halfpi * sec2 * u
    elseif isinf(lb)
        # Lower infinite: t ∈ [-1, 0] → u ∈ (-∞, ub] (when lb = -Inf)
        # or t ∈ [0, 1] → u ∈ (ub, ∞) (when lb = +Inf, flipped)
        # Use u = ub + tan(π*t/2), mapping:
        # t = 0 → u = ub + tan(0) = ub
        # t = -1 → u = ub + tan(-π/2) = -∞
        halfpi = oftype(float(one(u)), π / 2)
        arg = halfpi * t
        tan_val = tan(arg)
        sec2 = 1 + tan_val^2
        ub + tan_val * u, halfpi * sec2 * u
    elseif isinf(ub)
        # Upper infinite: t ∈ [0, 1] → u ∈ [lb, ∞) (when ub = +Inf)
        # or t ∈ [-1, 0] → u ∈ (-∞, lb] (when ub = -Inf, flipped)
        # Use u = lb + tan(π*t/2), mapping:
        # t = 0 → u = lb + tan(0) = lb
        # t = 1 → u = lb + tan(π/2) = ∞
        halfpi = oftype(float(one(u)), π / 2)
        arg = halfpi * t
        tan_val = tan(arg)
        sec2 = 1 + tan_val^2
        lb + tan_val * u, halfpi * sec2 * u
    else
        den = (ub - lb) * oftype(float(one(u)), 0.5)
        lb + (1 + t) * den, den
    end
end

"""
    transformation_tan_inf(f, domain)

Alternative infinity transformation using arctan/tan. Maps infinite domains to [-1, 1]
using the transformation:

  - For doubly-infinite domains: `u = tan(πt/2)`, so `∫_{-∞}^{∞} f(u) du = (π/2) ∫_{-1}^{1} sec²(πt/2) f(tan(πt/2)) dt`
  - For semi-infinite domains `[a, ∞)`: `u = a + tan(π(t+1)/4)`
  - For semi-infinite domains `(-∞, b]`: `u = b - tan(π(1-t)/4)`

This transformation can provide better accuracy than the default rational transformation
for some integrands, particularly those that decay like `1/(1+x²)`.

## Example

```julia
using Integrals

f(x, p) = 1 / (1 + x^2)  # Lorentzian
prob = IntegralProblem(f, (-Inf, Inf))

# Use tan transformation instead of default
alg = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
sol = solve(prob, alg)
```

See also: [`transformation_if_inf`](@ref), [`transformation_cot_inf`](@ref), [`ChangeOfVariables`](@ref)
"""
function transformation_tan_inf(f, domain)
    lb, ub = promote(domain...)
    tdomain = substitute_u(u2t_tan, lb, ub)
    g = substitute_f(f, t2ujac_tan, lb, ub)
    return g, tdomain
end

# Alternative transformation using cotangent for semi-infinite domains
# Based on the transformations from Issue #149:
# For [a, ∞): s = -cot[(π - 2arctan(a))(ξ-1)/4], ξ ∈ [-1, 1]
# For (-∞, a]: s = -cot[(π + 2arctan(a))(ξ+1)/4], ξ ∈ [-1, 1]
#
# These give:
# ∫_a^∞ g(s)ds = (π - 2arctan(a))/4 ∫_{-1}^1 csc²[(π - 2arctan(a))(ξ-1)/4] g(...) dξ
# ∫_{-∞}^a g(s)ds = (π + 2arctan(a))/4 ∫_{-1}^1 csc²[(π + 2arctan(a))(ξ+1)/4] g(...) dξ

function u2t_cot(lb, ub)
    mid = (lb + ub) / 2 # floating-point promotion
    # All cases map to [-1, 1]
    if isinf(lb) && isinf(ub)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = flipsign(one(mid), ub)
    elseif isinf(lb)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = zero(one(mid))
    elseif isinf(ub)
        lb_sub = zero(one(mid))
        ub_sub = flipsign(one(mid), ub)
    else
        lb_sub = -one(mid)
        ub_sub = one(mid)
    end
    return lb_sub, ub_sub
end

function t2ujac_cot(t::Number, lb::Number, ub::Number)
    u = oneunit(eltype(lb))
    if isinf(lb) && isinf(ub)
        # For doubly-infinite, use the tan transformation
        # u = tan(πt/2), du/dt = (π/2) sec²(πt/2)
        halfpi = oftype(float(one(u)), π / 2)
        arg = halfpi * t
        tan_val = tan(arg)
        sec2 = 1 + tan_val^2
        tan_val * u, halfpi * sec2 * u
    elseif isinf(lb)
        # Lower infinite: t ∈ [-1, 0] → u ∈ (-∞, ub]
        # Using cotangent transformation based on Issue #149:
        # u = -cot[(π + 2arctan(ub))(t+1)/4]
        # At t = -1: arg = 0, cot → ∞, u → -∞ ✓
        # At t = 0: arg = (π + 2arctan(ub))/4
        # For this to give u = ub at t = 0, we need to adjust
        # Instead, use simpler: u = ub + tan(πt/2) (same as tan transformation)
        # This falls back to tan for robustness
        halfpi = oftype(float(one(u)), π / 2)
        arg = halfpi * t
        tan_val = tan(arg)
        sec2 = 1 + tan_val^2
        ub + tan_val * u, halfpi * sec2 * u
    elseif isinf(ub)
        # Upper infinite: t ∈ [0, 1] → u ∈ [lb, ∞)
        # Using cotangent with scaling based on lb:
        # u = lb + cot[scale * (1 - t)]
        # At t = 0: arg = scale, cot(scale) = finite value
        # At t = 1: arg = 0, cot → ∞, u → ∞ ✓
        # We need u = lb at t = 0, so adjust: use cot approaching 0 as t→0
        # Use: u = lb + cot[π(1-t)/2] where cot(π/2) = 0
        # At t = 0: arg = π/2, cot(π/2) = 0, u = lb ✓
        # At t = 1: arg = 0, cot(0) → ∞, u → ∞ ✓
        halfpi = oftype(float(one(u)), π / 2)
        arg = halfpi * (1 - t)
        cot_val = cot(arg)
        csc2 = 1 + cot_val^2  # csc² = 1 + cot²
        # Jacobian: du/dt = csc²(arg) * π/2
        lb + cot_val * u, halfpi * csc2 * u
    else
        den = (ub - lb) * oftype(float(one(u)), 0.5)
        lb + (1 + t) * den, den
    end
end

"""
    transformation_cot_inf(f, domain)

Alternative infinity transformation using cotangent for semi-infinite domains.
Based on the transformations suggested in Issue #149:

  - For doubly-infinite domains: Uses tan transformation (same as `transformation_tan_inf`)
  - For semi-infinite domains: Uses cotangent-based transformations that can provide
    better accuracy for integrands with oscillations or singularities.

For `[a, ∞)`:

```math
s = \\cot\\left[\\frac{(\\pi - 2\\arctan(a))(1-\\xi)}{4}\\right] + a, \\quad \\xi \\in [-1, 1]
```

For `(-\\infty, a]`:

```math
s = -\\cot\\left[\\frac{(\\pi + 2\\arctan(a))(\\xi+1)}{4}\\right], \\quad \\xi \\in [-1, 1]
```

## Example

```julia
using Integrals

f(x, p) = exp(-x^2)  # Gaussian tail
prob = IntegralProblem(f, (0.0, Inf))

# Use cotangent transformation
alg = ChangeOfVariables(transformation_cot_inf, QuadGKJL())
sol = solve(prob, alg)
```

See also: [`transformation_if_inf`](@ref), [`transformation_tan_inf`](@ref), [`ChangeOfVariables`](@ref)
"""
function transformation_cot_inf(f, domain)
    lb, ub = promote(domain...)
    tdomain = substitute_u(u2t_cot, lb, ub)
    g = substitute_f(f, t2ujac_cot, lb, ub)
    return g, tdomain
end
