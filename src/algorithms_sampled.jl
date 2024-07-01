abstract type AbstractSampledIntegralAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

"""
    TrapezoidalRule

Struct for evaluating an integral via the trapezoidal rule.

Example with sampled data:

```julia
using Integrals
f = x -> x^2
x = range(0, 1, length=20)
y = f.(x)
problem = SampledIntegralProblem(y, x)
method = TrapezoidalRule()
solve(problem, method)
```
"""
struct TrapezoidalRule <: AbstractSampledIntegralAlgorithm end

"""
    SimpsonsRule

Struct for evaluating an integral via the Simpson's composite 1/3-3/8
rule over `AbstractRange`s (evenly spaced points) and
Simpson's composite 1/3 rule for non-equidistant grids.

Example with equidistant data:

```julia
using Integrals
f = x -> x^2
x = range(0, 1, length=20)
y = f.(x)
problem = SampledIntegralProblem(y, x)
method = SimpsonsRule()
solve(problem, method)
```
"""
struct SimpsonsRule <: AbstractSampledIntegralAlgorithm end
