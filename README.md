# Integrals.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/Integrals/stable/)

[![codecov](https://codecov.io/gh/SciML/Integrals.jl/branch/master/graph/badge.svg)](https://app.codecov.io/gh/SciML/Integrals.jl)
[![Build Status](https://github.com/SciML/Integrals.jl/workflows/CI/badge.svg)](https://github.com/SciML/Integrals.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
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
[see the stable documentation](https://docs.sciml.ai/Integrals/stable/). Use the
[in-development documentation](https://docs.sciml.ai/Integrals/dev/) for the version of
the documentation, which contains the unreleased features.

## Examples

To perform one-dimensional quadrature, we can simply construct an `IntegralProblem`. The code below evaluates $\int_{-2}^5 \sin(xp)~\mathrm{d}x$ with $p = 1.7$. This argument $p$ is passed
into the problem as the third argument of `IntegralProblem`.

```julia
using Integrals
f(x, p) = sin(x * p)
p = 1.7
domain = (-2, 5) # (lb, ub)
prob = IntegralProblem(f, domain, p)
sol = solve(prob, QuadGKJL())
```

For basic multidimensional quadrature we can construct and solve a `IntegralProblem`. Since we are using no arguments `p` in this example, we omit the third argument of `IntegralProblem`
from above. The lower and upper bounds are now passed as vectors, with the `i`th elements of
the bounds giving the interval of integration for `x[i]`.

```julia
using Integrals
f(x, p) = sum(sin.(x))
domain = (ones(2), 3ones(2)) # (lb, ub)
prob = IntegralProblem(f, domain)
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
```

If we would like to parallelize the computation, we can use the batch interface
to compute multiple points at once. For example, here we do allocation-free
multithreading with Cubature.jl:

```julia
using Integrals, Cubature, Base.Threads
function f(dx, x, p)
    Threads.@threads for i in 1:size(x, 2)
        dx[i] = sum(sin, @view(x[:, i]))
    end
end
domain = (ones(2), 3ones(2)) # (lb, ub)
prob = IntegralProblem(BatchIntegralFunction(f, zeros(0)), domain)
sol = solve(prob, CubatureJLh(), reltol = 1e-3, abstol = 1e-3)
```

If we would like to compare the results against Cuba.jl's `Cuhre` method, then
the change is a one-argument change:

```julia
using Cuba
sol = solve(prob, CubaCuhre(), reltol = 1e-3, abstol = 1e-3)
```
