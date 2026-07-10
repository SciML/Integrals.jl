"""
    AbstractSampledIntegralAlgorithm <: SciMLBase.AbstractIntegralAlgorithm

Abstract type for integration algorithms that work with sampled data points,
such as the trapezoidal rule and Simpson's rule.

## Interface

Concrete subtypes are used with `SampledIntegralProblem` and must support
`find_weights(x, alg)`, where `x` is the problem sampling grid and `alg` is the
concrete algorithm. The returned weights must:

  - have the same length as the integration axis of the sampled data,
  - support `iterate`, `length`, `eltype`, `size`, and scalar indexing, and
  - be valid inputs to `evalrule(data, weights, dim)`.

The generic sampled solver calls `find_weights` during `init` and recomputes weights
when the cache grid `x` is replaced.
"""
abstract type AbstractSampledIntegralAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

"""
    TrapezoidalRule

Composite trapezoidal rule for integrating sampled data.

## Returns

Returns a `TrapezoidalRule` algorithm object for
`solve(prob::SampledIntegralProblem, alg)`.

## Example

```julia
using Integrals

f = x -> x^2
x = range(0, 1, length = 20)
y = f.(x)
prob = SampledIntegralProblem(y, x)
sol = solve(prob, TrapezoidalRule())
```
"""
struct TrapezoidalRule <: AbstractSampledIntegralAlgorithm end

"""
    SimpsonsRule

Composite Simpson rule for integrating sampled data.

For evenly spaced `AbstractRange` grids this uses the composite Simpson 1/3 and 3/8
rules. For non-equidistant grids it uses a composite Simpson 1/3 construction.

## Returns

Returns a `SimpsonsRule` algorithm object for
`solve(prob::SampledIntegralProblem, alg)`.

## Example

```julia
using Integrals

f = x -> x^2
x = range(0, 1, length = 21)
y = f.(x)
prob = SampledIntegralProblem(y, x)
sol = solve(prob, SimpsonsRule())
```
"""
struct SimpsonsRule <: AbstractSampledIntegralAlgorithm end
