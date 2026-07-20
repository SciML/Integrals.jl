# Common Solver Options (Solve Keyword Arguments)

```@docs
solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
```

## Related SciML interface utilities

```@docs
init(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
init(prob::SampledIntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
solve!(cache::Integrals.IntegralCache)
solve!(cache::Integrals.SampledIntegralCache)
isinplace(cache::Integrals.IntegralCache)
ReturnCode
```
