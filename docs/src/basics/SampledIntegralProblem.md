# Integrating pre-sampled data

In some cases, instead of a function that acts as integrand,
one only possesses a list of data points `y` at a set of sampling
locations `x`, that must be integrated. This package contains functionality
for doing that.

## Example

Say, by some means we have generated a dataset `x` and `y`:

```@example 1
using Integrals # hide
f = x -> x^2
x = range(0, 1, length=20)
y = f.(x)
```

Now, we can integrate this data set as follows:

```@example 1
problem = SampledIntegralProblem(y, x)
method = TrapezoidalRule()
solve(problem, method)
```

The exact answer is of course \$ 1/3 \$.

## Details

```@docs
SciMLBase.SampledIntegralProblem
solve(::SampledIntegralProblem, ::SciMLBase.AbstractIntegralAlgorithm)
```

### Non-equidistant grids

If the sampling points `x` are provided as an `AbstractRange`
(constructed with the `range` function for example), faster methods are used that take advantage of
the fact that the points are equidistantly spaced. Otherwise, general methods are used for
non-uniform grids.

Example:

```@example 2
using Integrals # hide
f = x -> x^7
x = [0.0; sort(rand(1000)); 1.0]
y = f.(x)
problem = SampledIntegralProblem(y, x)
method = TrapezoidalRule()
solve(problem, method)
```

### Evaluating multiple integrals at once

If the provided data set `y` is a multidimensional array, the integrals are evaluated across only one
of its axes. For performance reasons, the last axis of the array `y` is chosen by default, but this can be modified with the `dim`
keyword argument to the problem definition.

```@example 3
using Integrals # hide
f1 = x -> x^2
f2 = x -> x^3
f3 = x -> x^4
x = range(0, 1, length=20)
y = [f1.(x) f2.(x) f3.(x)]
problem = SampledIntegralProblem(y, x; dim=1)
method = SimpsonsRule()
solve(problem, method)
```

### Supported methods

Right now, only the [`TrapezoidalRule`](https://en.wikipedia.org/wiki/Trapezoidal_rule) and [`SimpsonsRule`](https://en.wikipedia.org/wiki/Simpson%27s_rule) are supported.

```@docs
TrapezoidalRule
SimpsonsRule
```
