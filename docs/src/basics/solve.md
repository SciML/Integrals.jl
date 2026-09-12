# Common Solver Options (Solve Keyword Arguments)

`solve` is the generic SciMLBase interface reexported by Integrals.jl. Integrals.jl
implements it for integral problems and the algorithms documented on this site.

```@docs
solve(::IntegralProblem, ::SciMLBase.AbstractIntegralAlgorithm)
solve(::SampledIntegralProblem, ::SciMLBase.AbstractIntegralAlgorithm)
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
