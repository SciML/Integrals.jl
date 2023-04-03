struct IntegralCache{P, A, S, K, Tc}
    prob::P
    alg::A
    sensealg::S
    kwargs::K
    # cache for algorithm goes here (currently unused)
    cacheval::Tc
    isfresh::Bool
end

function SciMLBase.init(prob::IntegralProblem,
                        alg::SciMLBase.AbstractIntegralAlgorithm;
                        sensealg = ReCallVJP(ZygoteVJP()),
                        do_inf_transformation = nothing, kwargs...)
    checkkwargs(kwargs...)
    prob = transformation_if_inf(prob, do_inf_transformation)
    cacheval = nothing
    isfresh = true

    IntegralCache{typeof(prob),
                  typeof(alg),
                  typeof(sensealg),
                  typeof(kwargs),
                  typeof(cacheval)}(prob,
                                    alg,
                                    sensealg,
                                    kwargs,
                                    cacheval,
                                    isfresh)
end

# Throw error if alg is not provided, as defaults are not implemented.
function SciMLBase.solve(::IntegralProblem; kwargs...)
    checkkwargs(kwargs...)
    throw(ArgumentError("""
No integration algorithm `alg` was supplied as the second positional argument.
Reccomended integration algorithms are:
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

function SciMLBase.solve!(cache::IntegralCache)
    prob = cache.prob
    __solvebp(prob, cache.alg, cache.sensealg, prob.lb, prob.ub, prob.p; cache.kwargs...)
end
