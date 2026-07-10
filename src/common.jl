mutable struct IntegralCache{iip, F, D, P, PK, A, S, K, Tc, VT}
    iip::Val{iip}
    f::F
    domain::D
    p::P
    prob_kwargs::PK
    alg::A
    sensealg::S
    kwargs::K
    cacheval::Tc    # store alg cache here
    verbosity::VT
end

"""
    isinplace(cache::IntegralCache) -> Bool

Return whether an initialized integral cache wraps an in-place integrand.

## Arguments

  - `cache`: Cache returned by [`init`](@ref) for an `IntegralProblem`.

## Returns

Returns `true` when the cached integral function mutates its output argument, and
`false` otherwise.

## Example

```julia
using Integrals

prob = IntegralProblem((x, p) -> x^2, (0.0, 1.0))
cache = init(prob, QuadGKJL())
isinplace(cache)
```
"""
SciMLBase.isinplace(::IntegralCache{iip}) where {iip} = iip

init_cacheval(::SciMLBase.AbstractIntegralAlgorithm, args...) = nothing

"""
    init(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)

Initialize a reusable integral solver cache for `prob` and `alg`.

## Arguments

  - `prob`: Integral problem containing the integrand, domain, parameters, and problem
    keyword options.
  - `alg`: Integration algorithm, such as `QuadGKJL()`, `HCubatureJL()`, or another
    subtype of `SciMLBase.AbstractIntegralAlgorithm`.

## Keyword Arguments

  - `sensealg`: Sensitivity algorithm used by automatic differentiation paths. Defaults to
    `ReCallVJP(ZygoteVJP())`.
  - `verbose`: Verbosity control via [`IntegralVerbosity`](@ref). Defaults to
    `IntegralVerbosity()`.
  - `do_inf_transformation`: Deprecated. Infinite-domain transformations are always
    applied when needed.
  - Additional keyword arguments are forwarded to the eventual solve.

## Returns

Returns an internal cache object that stores the transformed algorithm, cached backend
state, verbosity settings, and solve keyword arguments for reuse with [`solve!`](@ref).

## Example

```julia
using Integrals

prob = IntegralProblem((x, p) -> sin(x), (0.0, pi))
cache = init(prob, QuadGKJL(); reltol = 1e-10)
solve!(cache)
```
"""
function SciMLBase.init(
        prob::IntegralProblem{iip},
        alg::SciMLBase.AbstractIntegralAlgorithm;
        sensealg = ReCallVJP(ZygoteVJP()),
        do_inf_transformation = nothing,
        verbose = IntegralVerbosity(),
        kws...
    ) where {iip}
    kwargs = pairs((; prob.kwargs..., kws...))

    checkkwargs(kwargs...)
    verb_spec = _process_verbose_param(verbose)

    do_inf_transformation === nothing ||
        @SciMLMessage(
        "do_inf_transformation is deprecated. All integral problems are transformed",
        verb_spec, :deprecations
    )
    _alg = if alg isa ChangeOfVariables
        alg
    elseif prob.domain isa Tuple
        ChangeOfVariables(transformation_if_inf, alg)
    else
        alg
    end

    cacheval = init_cacheval(_alg, prob)

    @SciMLMessage("Cache initialization complete", verb_spec, :cache_init)

    return IntegralCache{
        iip,
        typeof(prob.f),
        typeof(prob.domain),
        typeof(prob.p),
        typeof(prob.kwargs),
        typeof(_alg),
        typeof(sensealg),
        typeof(kwargs),
        typeof(cacheval),
        typeof(verb_spec),
    }(
        Val(iip),
        prob.f,
        prob.domain,
        prob.p,
        prob.kwargs,
        _alg,
        sensealg,
        kwargs,
        cacheval
        ,
        verb_spec
    )
end

function Base.getproperty(cache::IntegralCache, name::Symbol)
    if name === :lb
        domain = getfield(cache, :domain)
        lb, ub = domain
        return lb
    elseif name === :ub
        domain = getfield(cache, :domain)
        lb, ub = domain
        return ub
    end
    return getfield(cache, name)
end
function Base.setproperty!(cache::IntegralCache, name::Symbol, x)
    if name === :lb
        @SciMLMessage("updating lb is deprecated by replacing domain", cache.verbosity, :deprecations)
        lb, ub = cache.domain
        setfield!(cache, :domain, (oftype(lb, x), ub))
        return x
    elseif name === :ub
        @SciMLMessage("updating ub is deprecated by replacing domain", cache.verbosity, :deprecations)
        lb, ub = cache.domain
        setfield!(cache, :domain, (lb, oftype(ub, x)))
        return x
    end
    return setfield!(cache, name, x)
end

# Throw error if alg is not provided, as defaults are not implemented.
function SciMLBase.solve(::IntegralProblem; kwargs...)
    checkkwargs(kwargs...)
    throw(
        ArgumentError(
            """
            No integration algorithm `alg` was supplied as the second positional argument.
            Recommended integration algorithms are:
            For scalar functions: QuadGKJL()
            For ≤ 8 dimensional vector functions: HCubatureJL()
            For > 8 dimensional vector functions: MonteCarloIntegration.vegas(f, st, en, kwargs...)
            See the docstrings of the different algorithms for more detail.
            """
        )
    )
end

"""
    solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)

Solve an integral problem with the specified integration algorithm.

## Arguments

  - `prob`: Integral problem containing the integrand, domain, parameters, and problem
    keyword options.
  - `alg`: Integration algorithm, such as `QuadGKJL()`, `HCubatureJL()`, or another
    subtype of `SciMLBase.AbstractIntegralAlgorithm`.

## Keyword Arguments

The arguments to `solve` are common across all of the quadrature methods.
These common arguments are:

  - `maxiters`: Maximum number of algorithm iterations or backend evaluations.
  - `abstol`: Absolute tolerance for the integral residual/error estimate.
  - `reltol`: Relative tolerance for the integral residual/error estimate.
  - `verbose`: Verbosity control via [`IntegralVerbosity`](@ref). Defaults to
    `IntegralVerbosity()`.

## Returns

Returns a SciMLBase integral solution whose `u` field is the integral estimate and whose
`resid` field is the backend residual or error estimate when available.

## Example

```julia
using Integrals

prob = IntegralProblem((x, p) -> sin(x), (0.0, pi))
sol = solve(prob, QuadGKJL(); reltol = 1e-10, abstol = 1e-10)
```
"""
function SciMLBase.solve(
        prob::IntegralProblem,
        alg::SciMLBase.AbstractIntegralAlgorithm;
        kwargs...
    )
    return solve!(init(prob, alg; kwargs...))
end

"""
    solve!(cache::IntegralCache)

Solve an initialized integral cache.

## Arguments

  - `cache`: Cache returned by [`init`](@ref) for an `IntegralProblem`.

## Returns

Returns a SciMLBase integral solution whose `u` field is the integral estimate and whose
`resid` field is the backend residual or error estimate when available.

## Example

```julia
using Integrals

prob = IntegralProblem((x, p) -> x^2, (0.0, 1.0))
cache = init(prob, QuadGKJL())
sol = solve!(cache)
```
"""
function SciMLBase.solve!(cache::IntegralCache)
    return __solve(
        cache, cache.alg, cache.sensealg, cache.domain, cache.p;
        cache.kwargs...
    )
end

function build_problem(cache::IntegralCache{iip}) where {iip}
    return IntegralProblem{iip}(cache.f, cache.domain, cache.p; cache.prob_kwargs...)
end

# fallback method for existing algorithms which use no cache
function __solvebp_call(cache::IntegralCache, args...; kwargs...)
    return __solvebp_call(build_problem(cache), args...; verbose = cache.verbosity, kwargs...)
end

mutable struct SampledIntegralCache{Y, X, D, PK, A, K, Tc, VT}
    y::Y
    x::X
    dim::D
    prob_kwargs::PK
    alg::A
    kwargs::K
    isfresh::Bool   # state of whether weights have been calculated
    cacheval::Tc    # store alg weights here
    verbosity::VT
end

function Base.setproperty!(cache::SampledIntegralCache, name::Symbol, x)
    if name === :x
        setfield!(cache, :isfresh, true)
    end
    return setfield!(cache, name, x)
end

"""
    init(prob::SampledIntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)

Initialize a reusable sampled-data integral solver cache.

## Arguments

  - `prob`: Sampled integral problem containing sampled values, sampling points, and the
    integration dimension.
  - `alg`: Sampled-data integration algorithm, such as `TrapezoidalRule()` or
    `SimpsonsRule()`.

## Keyword Arguments

  - `verbose`: Verbosity control via [`IntegralVerbosity`](@ref). Defaults to
    `IntegralVerbosity()`.

No other keyword arguments are accepted by this cache initializer.

## Returns

Returns an internal cache object that can be reused with [`solve!`](@ref). Updating the
sample locations marks the cached weights stale so they are recomputed before the next
solve.

## Example

```julia
using Integrals

x = range(0, 1, length = 21)
prob = SampledIntegralProblem(x .^ 2, x)
cache = init(prob, SimpsonsRule())
solve!(cache)
```
"""
function SciMLBase.init(
        prob::SampledIntegralProblem,
        alg::SciMLBase.AbstractIntegralAlgorithm;
        verbose = IntegralVerbosity(),
        kwargs...
    )
    NamedTuple(kwargs) == NamedTuple() ||
        throw(ArgumentError("There are no keyword arguments allowed to `solve`"))

    verb_spec = _process_verbose_param(verbose)
    cacheval = init_cacheval(alg, prob)
    isfresh = true

    return SampledIntegralCache(
        prob.y,
        prob.x,
        prob.dim,
        prob.kwargs,
        alg,
        kwargs,
        isfresh,
        cacheval,
        verb_spec
    )
end

"""
    solve(prob::SampledIntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)

Integrate sampled data with a sampled-data quadrature algorithm.

## Arguments

  - `prob`: Sampled integral problem containing sampled values `y`, sampling points
    `x`, and the integration dimension.
  - `alg`: Sampled-data integration algorithm, such as `TrapezoidalRule()` or
    `SimpsonsRule()`.

## Keyword Arguments

There are no algorithm-independent keyword arguments used to solve
`SampledIntegralProblem`s.

## Returns

Returns a SciMLBase integral solution whose `u` field is the sampled integral estimate.

## Example

```julia
using Integrals

x = range(0, 1, length = 21)
prob = SampledIntegralProblem(x .^ 2, x)
sol = solve(prob, SimpsonsRule())
```
"""
function SciMLBase.solve(
        prob::SampledIntegralProblem,
        alg::SciMLBase.AbstractIntegralAlgorithm;
        kwargs...
    )
    return solve!(init(prob, alg; kwargs...))
end

"""
    solve!(cache::SampledIntegralCache)

Solve an initialized sampled-data integral cache.

## Arguments

  - `cache`: Cache returned by [`init`](@ref) for a `SampledIntegralProblem`.

## Returns

Returns a SciMLBase integral solution whose `u` field is the sampled integral estimate.

## Example

```julia
using Integrals

x = range(0, 1, length = 21)
prob = SampledIntegralProblem(x .^ 2, x)
cache = init(prob, TrapezoidalRule())
sol = solve!(cache)
```
"""
function SciMLBase.solve!(cache::SampledIntegralCache)
    return __solvebp(cache, cache.alg; cache.kwargs...)
end

function build_problem(cache::SampledIntegralCache)
    return SampledIntegralProblem(
        cache.y,
        cache.x;
        dim = dimension(cache.dim),
        cache.prob_kwargs...
    )
end
