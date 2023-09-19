mutable struct IntegralCache{iip, F, B, P, PK, A, S, K, Tc}
    iip::Val{iip}
    f::F
    lb::B
    ub::B
    nout::Int
    p::P
    batch::Int
    prob_kwargs::PK
    alg::A
    sensealg::S
    kwargs::K
    cacheval::Tc    # store alg cache here
end

SciMLBase.isinplace(::IntegralCache{iip}) where {iip} = iip

init_cacheval(::SciMLBase.AbstractIntegralAlgorithm, args...) = nothing

function SciMLBase.init(prob::IntegralProblem{iip},
    alg::SciMLBase.AbstractIntegralAlgorithm;
    sensealg = ReCallVJP(ZygoteVJP()),
    do_inf_transformation = nothing, kwargs...) where {iip}
    checkkwargs(kwargs...)
    prob = transformation_if_inf(prob, do_inf_transformation)
    cacheval = init_cacheval(alg, prob)

    IntegralCache{iip,
        typeof(prob.f),
        typeof(prob.lb),
        typeof(prob.p),
        typeof(prob.kwargs),
        typeof(alg),
        typeof(sensealg),
        typeof(kwargs),
        typeof(cacheval)}(Val(iip),
        prob.f,
        prob.lb,
        prob.ub,
        prob.nout,
        prob.p,
        prob.batch,
        prob.kwargs,
        alg,
        sensealg,
        kwargs,
        cacheval)
end

# Throw error if alg is not provided, as defaults are not implemented.
function SciMLBase.solve(::IntegralProblem; kwargs...)
    checkkwargs(kwargs...)
    throw(ArgumentError("""
No integration algorithm `alg` was supplied as the second positional argument.
Recommended integration algorithms are:
For scalar functions: QuadGKJL()
For ≤ 8 dimensional vector functions: HCubatureJL()
For > 8 dimensional vector functions: MonteCarloIntegration.vegas(f, st, en, kwargs...)
See the docstrings of the different algorithms for more detail.
"""))
end

"""
```julia
solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)
```

## Keyword Arguments

The arguments to `solve` are common across all of the quadrature methods.
These common arguments are:

  - `maxiters` (the maximum number of iterations)
  - `abstol` (absolute tolerance in changes of the objective value)
  - `reltol` (relative tolerance  in changes of the objective value)
"""
function SciMLBase.solve(prob::IntegralProblem,
    alg::SciMLBase.AbstractIntegralAlgorithm;
    kwargs...)
    solve!(init(prob, alg; kwargs...))
end

function SciMLBase.solve(prob::SampledIntegralProblem,
    alg::SciMLBase.AbstractIntegralAlgorithm;
    kwargs...)
    __solvebp(prob, alg; kwargs...)
end

function SciMLBase.solve!(cache::IntegralCache)
    __solvebp(cache, cache.alg, cache.sensealg, cache.lb, cache.ub, cache.p;
        cache.kwargs...)
end

function build_problem(cache::IntegralCache{iip}) where {iip}
    IntegralProblem{iip}(cache.f, cache.lb, cache.ub, cache.p;
        nout = cache.nout, batch = cache.batch, cache.prob_kwargs...)
end

# fallback method for existing algorithms which use no cache
function __solvebp_call(cache::IntegralCache, args...; kwargs...)
    __solvebp_call(build_problem(cache), args...; kwargs...)
end

