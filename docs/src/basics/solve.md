# Common Solver Options (Solve Keyword Arguments)

```@docs
solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm)
```

Additionally, the extra keyword arguments are splatted to the library calls, so
see the documentation of the integrator library for all of the extra details.
These extra keyword arguments are not guaranteed to act uniformly.