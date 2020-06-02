# Quadrature.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/Quadrature.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/Quadrature.jl)

Quadrature.jl is an instantiation of the DiffEqBase.jl common `QuadratureProblem`
interface for the common quadrature packages of Julia. By using Quadrature.jl,
you get a single predictable interface where many of the arguments are
standardized throughout the various integrator libraries. This can be useful
for benchmarking or for library implementations, since libraries which internally
use a quadrature can easily accept a quadrature method as an argument.

## Examples

For basic multidimensional quadrature we can construct and solve a `QuadratureProblem`:

```julia
using Quadrature
f(x,p) = sum(sin.(x))
prob = QuadratureProblem(f,ones(2),3ones(2))
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
```

If we would like to parallelize the computation, we can use the batch interface
to compute multiple points at once. For example, here we do allocation-free
multithreading with Cubature.jl:

```julia
using Quadrature, Cubature, Base.Threads
function f(dx,x,p)
  Threads.@threads for i in 1:size(x,2)
    dx[i] = sum(sin.(@view(x[:,i])))
  end
end
prob = QuadratureProblem(f,ones(2),3ones(2),batch=2)
sol = solve(prob,CubatureJLh(),reltol=1e-3,abstol=1e-3)
```

If we would like to compare the results against Cuba.jl's `Cuhre` method, then
the change is one argument change:

```julia
using Cuba
sol = solve(prob,CubaCuhre(),reltol=1e-3,abstol=1e-3)
```

## QuadratureProblem

To use this package, you always construct a `QuadratureProblem`. This has a
constructor:

```julia
QuadratureProblem(f,lb,ub,p=NullParameters();
                  nout=1, batch = 0, kwargs...)
```

- `f`: Either a function `f(x,p)` for out-of-place or `f(dx,x,p)` for in place.
- `lb`: Either a number or vector of lower bounds
- `ub`: Either a number or vector of upper bounds
- `p`: The parameters associated with the problem
- `nout`: The output size of the function `f`. Defaults to `1`, i.e. a scalar
  integral output.
- `batch`: The preferred number of points to batch. This allows user-side
  parallelization of the integrand. If `batch != 0`, then each `x[:,i]` is a
  different point of the integral to calculate, and the output should be
  `nout x batchsize`. Note that `batch` is a suggestion for the number of points,
  and it is not necessarily true that `batch` is the same as `batchsize` in all
  algorithms.

Additionally, we can supply `iip` like `QuadratureProblem{iip}(...)` as true or
false to declare at compile time whether the integrator function is in-place.

## Algorithms

The following algorithms are available:

- `QuadGKJL`: Uses QuadGK.jl. Requires `nout=1` and `batch=0`.
- `HCubatureJL`: Uses HCubature.jl. Requires `batch=0`.
- `VEGAS`: Uses MonteCarloIntegration.jl. Requires `nout=1`.
- `CubatureJLh`: h-Cubature from Cubature.jl. Requires `using Cubature`.
- `CubatureJLp`: p-Cubature from Cubature.jl. Requires `using Cubature`.
- `CubaVegas`: Vegas from Cuba.jl. Requires `using Cuba`.
- `CubaSUAVE`: SUAVE from Cuba.jl. Requires `using Cuba`.
- `CubaDivonne`: Divonne from Cuba.jl. Requires `using Cuba`.
- `CubaCuhre`: Cuhre from Cuba.jl. Requires `using Cuba`.

## Common Solve Keyword Arguments

- `reltol`: Relative tolerance
- `abstol`: Absolute tolerance
- `maxiters`: The maximum number of iterations

Additionally, the extra keyword arguments are splatted to the library calls, so
see the documentation of the integrator library for all of the extra details.
These extra keyword arguments are not guaranteed to act uniformly.
