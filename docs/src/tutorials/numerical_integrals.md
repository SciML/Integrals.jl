# Numerically Solving Integrals

For basic multidimensional quadrature, we can construct and solve a `IntegralProblem`.
The integral we want to evaluate is:

```math
\int_1^3\int_1^3\int_1^3 \sum_1^3 \sin(u_i) du_1du_2du_3.
```

We can numerically approximate this integral:

```@example integrate1
using Integrals
f(u, p) = sum(sin.(u))
prob = IntegralProblem(f, ones(3), 3ones(3))
sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
sol.u
```

where the first argument of `IntegralProblem` is the integrand,
the second argument is the lower bound, and the third argument is the upper bound.
`p` are the parameters of the integrand. In this case, there are no parameters,
but still `f` must be defined as `f(x,p)` and **not** `f(x)`.
For an example with parameters, see the next tutorial.
The first argument of `solve` is the problem we are solving,
the second is an algorithm to solve the problem with.
Then there are keywords which provides details how the algorithm should work,
in this case tolerances how precise the numerical approximation should be.

We can also evaluate multiple integrals at once.
We could create two `IntegralProblem`s for this,
but that is wasteful if the integrands share a lot of computation.
For example, we also want to evaluate:

```math
\int_1^3\int_1^3\int_1^3 \sum_1^3 \cos(u_i) du_1du_2du_3.
```

```@example integrate2
using Integrals
f(u, p) = [sum(sin.(u)), sum(cos.(u))]
prob = IntegralProblem(f, ones(3), 3ones(3))
sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
sol.u
```

Another way to think about this is that the integrand is now a vector valued function.
In general, we should be able to integrate any type that is in a vector space
and supports addition and scalar multiplication, although Integrals.jl allows
scalars and arrays.
In the above example, the integrand was defined out-of-position.
This means that a new output vector is created every time the function `f` is called.
If we do not  want these allocations, we can also define `f` in-position.

```@example integrate3
using Integrals, Cubature
function f(y, u, p)
    y[1] = sum(sin.(u))
    y[2] = sum(cos.(u))
end
prototype = zeros(2)
prob = IntegralProblem(IntegralProblem(f, prototype), ones(3), 3ones(3))
sol = solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)
sol.u
```

where `y` is a cache to store the evaluation of the integrand and `prototype` is
an instance of `y` with the desired type and shape.
We needed to change the algorithm to `CubatureJLh()`
because `HCubatureJL()` does not support in-position under the hood.
`f` evaluates the integrand at a certain point,
but most adaptive quadrature algorithms need to evaluate the integrand at multiple points
in each step of the algorithm.
We would thus often like to parallelize the computation.
The batch interface allows us to compute multiple points at once.
For example, here we do allocation-free multithreading with Cubature.jl:

```@example integrate4
using Integrals, Cubature, Base.Threads
function f(y, u, p)
    Threads.@threads for i in 1:size(u, 2)
        y[1, i] = sum(sin, @view(u[:, i]))
        y[2, i] = sum(cos, @view(u[:, i]))
    end
end
prototype = zeros(2, 0)
prob = IntegralProblem(BatchIntegralFunction(f, prototype), ones(3), 3ones(3))
sol = solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)
sol.u
```

Both `u` and `y` changed from vectors to matrices,
where each column is respectively a point the integrand is evaluated at or
the evaluation of the integrand at the corresponding point.
The `prototype` now has an extra dimension for batching that can be of size zero.
Try to create yourself an out-of-position version of the above problem.
For the full details of the batching interface, see the [problem page](@ref prob).

If we would like to compare the results against Cuba.jl's `Cuhre` method, then
the change is a one-argument change:

```@example integrate5
using Integrals
using Cuba
f(u, p) = sum(sin.(u))
prob = IntegralProblem(f, ones(3), 3ones(3))
sol = solve(prob, CubaCuhre(); reltol = 1e-3, abstol = 1e-3)
sol.u
```

However, `Cuhre` does not support vector valued integrands.
The [solvers page](@ref solvers) gives an overview of which arguments each algorithm can handle.

## One-dimensional integrals

Integrals.jl also has specific solvers for integrals in a single dimension, such as `QuadGKJL`.
For example, we can create our own sine function by integrating the cosine function from 0 to x.

```@example integrate6
using Integrals
my_sin(x) = solve(IntegralProblem((x, p) -> cos(x), 0.0, x), QuadGKJL()).u
x = 0:0.1:(2 * pi)
@. my_sin(x) ≈ sin(x)
```

## Infinity handling

Integrals.jl can also handle infinite integration bounds.
For infinite upper bounds $u$ is substituted with $a+\frac{t}{1-t}$,
and the integral is thus transformed to:

```math
\int_a^\infty f(u)du = \int_0^1 f\left(a+\frac{t}{1-t}\right)\frac{1}{(1-t)^2}dt
```

Integrals with an infinite lower bound are handled in the same way.
If both upper and lower bound are infinite, $u$ is substituted with $\frac{t}{1-t^2}$,

```math
\int_{-\infty}^\infty f(u)du = \int_{-1}^1 f\left(\frac{t}{1-t^2}\right)\frac{1+t^2}{(1-t^2)^2}dt
```

For multidimensional integrals, each variable with infinite bounds is substituted the same way.
The details of the math behind these transforms can be found
[here](https://en.wikipedia.org/wiki/Integration_by_substitution#Substitution_for_multiple_variables).

As an example, let us integrate the standard bivariate normal probability distribution
over the area above the horizontal axis, which should be equal to $0.5$.

```@example integrate6
using Distributions
using Integrals
dist = MvNormal(ones(2))
f = (x, p) -> pdf(dist, x)
(lb, ub) = ([-Inf, 0.0], [Inf, Inf])
prob = IntegralProblem(f, lb, ub)
solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
```
