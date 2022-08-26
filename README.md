# Integrals.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://integrals.sciml.ai/stable/)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/dev/modules/Integrals/)

[![codecov](https://codecov.io/gh/SciML/Integrals.jl/branch/master/graph/badge.svg)](https://app.codecov.io/gh/SciML/Integrals.jl)
[![Build Status](https://github.com/SciML/Integrals.jl/workflows/CI/badge.svg)](https://github.com/SciML/Integrals.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Integrals.jl is an instantiation of the SciML common `IntegralProblem`
interface for the common numerical integration packages of Julia, including
both those based upon quadrature as well as Monte-Carlo approaches. By using
Integrals.jl, you get a single predictable interface where many of the
arguments are standardized throughout the various integrator libraries. This
can be useful for benchmarking or for library implementations, since libraries
which internally use a quadrature can easily accept a integration method as an
argument.

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
