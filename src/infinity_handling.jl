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
        _y = _evaluate!(f, prototype, u, p)
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

function transformation_if_inf(f, domain)
    lb, ub = promote(domain...)
    tdomain = substitute_u(u2t, lb, ub)
    g = substitute_f(f, t2ujac, lb, ub)
    return g, tdomain
end

# to implement more transformations, define functions u2v and v2ujac and then a wrapper
# similar to transformation_if_inf to pass to ChangeOfVariables
