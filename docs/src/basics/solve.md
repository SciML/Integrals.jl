# Common Solver Options (Solve Keyword Arguments)

`solve` is the generic SciMLBase interface reexported by Integrals.jl. Integrals.jl
implements it for integral problems and the algorithms documented on this site.

```@docs
solve(::IntegralProblem, ::SciMLBase.AbstractIntegralAlgorithm)
solve(::SampledIntegralProblem, ::SciMLBase.AbstractIntegralAlgorithm)
```

## Default tolerances

`QuadGKJL`, `HCubatureJL`, `CubatureJLh`, `CubatureJLp`, and
`HAdaptiveIntegrationJL` default to `abstol = 0` and `reltol = 1e-8`. With
`abstol = 0` the solvers control the integral residual through the relative
tolerance alone, so a small integral is not treated as converged just because its
residual is below a fixed absolute floor.

For integrands dominated by cancellation — large positive and negative
contributions that nearly cancel, such as the zero integral of `sin(2πx)` on
`[0, 1]` — a zero `abstol` can drive the solver to refine until it exhausts
`maxiters` without satisfying the relative bound. In that case pass a positive,
problem-scaled `abstol`, for example proportional to the magnitude of the
integrand times the domain volume:

```julia
sol = solve(prob, QuadGKJL(); abstol = 1e-8)
```

## Related SciML interface utilities

`init`, `solve!`, `isinplace`, `remake`, and `ReturnCode` are SciMLBase interface
utilities reexported by Integrals.jl. Their API is documented by SciMLBase.

```@docs
CommonSolve.init(::IntegralProblem{iip}, ::SciMLBase.AbstractIntegralAlgorithm) where {iip}
CommonSolve.init(::SampledIntegralProblem, ::SciMLBase.AbstractIntegralAlgorithm)
solve!(::Integrals.IntegralCache)
solve!(::Integrals.SampledIntegralCache)
isinplace(::Integrals.IntegralCache)
```
