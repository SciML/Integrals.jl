# Common Solver Options (Solve Keyword Arguments)

`solve` is the generic SciMLBase interface reexported by Integrals.jl. Integrals.jl
implements it for integral problems and the algorithms documented on this site.

```@docs
solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
solve(prob::SampledIntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
```

## Related SciML interface utilities

`init`, `solve!`, `isinplace`, `remake`, and `ReturnCode` are SciMLBase interface
utilities reexported by Integrals.jl. Their API is documented by SciMLBase.

```@docs
init(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
init(prob::SampledIntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
solve!(cache::Integrals.IntegralCache)
solve!(cache::Integrals.SampledIntegralCache)
isinplace(cache::Integrals.IntegralCache)
```
