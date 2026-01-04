# Infinite Domain Transformations

When integrating over infinite domains, Integrals.jl automatically applies a change of
variables to map the infinite interval to a finite one. This page explains the available
transformations and how to choose between them.

## Background

Many numerical integration algorithms (like Gauss-Kronrod quadrature) are designed for
finite intervals. To handle infinite domains, a substitution is applied:

```math
\int_a^\infty f(u) du = \int_{\text{finite}} g(v) dv
```

where the transformation `u = h(v)` maps the finite interval to the infinite one, and
`g(v) = f(h(v)) \cdot h'(v)` includes the Jacobian of the transformation.

The choice of transformation can significantly affect accuracy, especially for integrands
with specific decay behavior.

## Available Transformations

Integrals.jl provides three built-in transformations for handling infinite bounds:

### Default: `transformation_if_inf`

The default transformation uses rational functions. This is applied automatically when
using algorithms like `QuadGKJL` or `HCubatureJL` with infinite bounds.

```@docs
transformation_if_inf
```

### Alternative: `transformation_tan_inf`

Uses trigonometric (arctan/tan) transformation. This can work better for integrands
that decay like `1/(1+x²)`.

```@docs
transformation_tan_inf
```

### Alternative: `transformation_cot_inf`

Uses cotangent-based transformation for semi-infinite domains. This can be useful for
integrands with specific singularity behavior at the finite endpoint.

```@docs
transformation_cot_inf
```

## Using Custom Transformations

To use an alternative transformation, wrap your algorithm with `ChangeOfVariables`:

```@docs
Integrals.ChangeOfVariables
```

## Comparison Example

Let's compare the different transformations on a Gaussian integral:

```@example transforms
using Integrals

# Gaussian function
f(x, p) = exp(-x^2)
prob = IntegralProblem(f, (-Inf, Inf))

# True value: √π
true_value = sqrt(π)

# Default algorithm (uses transformation_if_inf internally)
sol_default = solve(prob, QuadGKJL())
println("Default:     ", sol_default.u, " (error: ", abs(sol_default.u - true_value), ")")

# Explicit default transformation
alg_default = ChangeOfVariables(transformation_if_inf, QuadGKJL())
sol_explicit = solve(prob, alg_default)
println("Explicit if: ", sol_explicit.u, " (error: ", abs(sol_explicit.u - true_value), ")")

# Tan transformation
alg_tan = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
sol_tan = solve(prob, alg_tan)
println("Tan:         ", sol_tan.u, " (error: ", abs(sol_tan.u - true_value), ")")

# Cot transformation
alg_cot = ChangeOfVariables(transformation_cot_inf, QuadGKJL())
sol_cot = solve(prob, alg_cot)
println("Cot:         ", sol_cot.u, " (error: ", abs(sol_cot.u - true_value), ")")
```

## Semi-Infinite Integrals

For semi-infinite integrals, the choice of transformation can be more important:

```@example transforms
# Integrating exp(-x) from 0 to ∞ (true value = 1)
f(x, p) = exp(-x)
prob = IntegralProblem(f, (0.0, Inf))

sol_default = solve(prob, QuadGKJL())
println("Default: ", sol_default.u)

alg_tan = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
sol_tan = solve(prob, alg_tan)
println("Tan:     ", sol_tan.u)

alg_cot = ChangeOfVariables(transformation_cot_inf, QuadGKJL())
sol_cot = solve(prob, alg_cot)
println("Cot:     ", sol_cot.u)
```

## When to Use Alternative Transformations

- **`transformation_if_inf` (default)**: Good general-purpose choice, works well for most
  smooth, rapidly decaying integrands.

- **`transformation_tan_inf`**: Consider when the integrand decays like `1/(1+x²)` or has
  polynomial-like tails. The arctan transformation naturally matches this decay behavior.

- **`transformation_cot_inf`**: Consider for semi-infinite integrals where the integrand
  has specific behavior near the finite boundary, or when other transformations show
  poor convergence.

## Implementing Custom Transformations

You can implement your own transformation by creating a function with signature:

```julia
my_transformation(f, domain) -> (g, new_domain)
```

where:
- `f` is the `IntegralFunction` or `BatchIntegralFunction`
- `domain` is a tuple `(lb, ub)` of bounds
- `g` is the transformed integral function
- `new_domain` is the new finite bounds

Then use it via:

```julia
alg = ChangeOfVariables(my_transformation, QuadGKJL())
```

For implementation details, see the source code of the built-in transformations in
`src/infinity_handling.jl`.
