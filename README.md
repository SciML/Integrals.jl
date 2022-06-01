# Integrals.jl

[![Build Status](https://github.com/SciML/Integrals.jl/workflows/CI/badge.svg)](https://github.com/SciML/Integrals.jl/actions?query=workflow%3ACI)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://integrals.sciml.ai/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://integrals.sciml.ai/dev/)

Integrals.jl is an instantiation of the SciML common `IntegralProblem`
interface for the common quadrature packages of Julia. By using Integrals.jl,
you get a single predictable interface where many of the arguments are
standardized throughout the various integrator libraries. This can be useful
for benchmarking or for library implementations, since libraries which internally
use a quadrature can easily accept a quadrature method as an argument.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://integrals.sciml.ai/stable/). Use the
[in-development documentation](https://integrals.sciml.ai/dev/) for the version of
the documentation, which contains the unreleased features.

## Examples

For basic multidimensional quadrature we can construct and solve a `IntegralProblem`:

```julia
using Integrals
f(x,p) = sum(sin.(x))
prob = IntegralProblem(f,ones(2),3ones(2))
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)
```

If we would like to parallelize the computation, we can use the batch interface
to compute multiple points at once. For example, here we do allocation-free
multithreading with Cubature.jl:

```julia
using Integrals, Cubature, Base.Threads
function f(dx,x,p)
  Threads.@threads for i in 1:size(x,2)
    dx[i] = sum(sin.(@view(x[:,i])))
  end
end
prob = IntegralProblem(f,ones(2),3ones(2),batch=2)
sol = solve(prob,CubatureJLh(),reltol=1e-3,abstol=1e-3)
```

If we would like to compare the results against Cuba.jl's `Cuhre` method, then
the change is a one-argument change:

```julia
using IntegralsCuba
sol = solve(prob,CubaCuhre(),reltol=1e-3,abstol=1e-3)
```