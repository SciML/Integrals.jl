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

The default transformation uses rational functions to map infinite domains to finite intervals:

- For doubly-infinite domains `(-∞, ∞)`: uses `u = t/(1-t²)` mapping `[-1, 1] → (-∞, ∞)`
- For semi-infinite domains `[a, ∞)`: uses `u = a + t/(1-t)` mapping `[0, 1] → [a, ∞)`
- For semi-infinite domains `(-∞, b]`: uses `u = b + t/(1+t)` mapping `[-1, 0] → (-∞, b]`

This transformation is applied automatically when using algorithms like `QuadGKJL()` or `HCubatureJL()` with infinite bounds.

### Alternative: `transformation_tan_inf`

Uses trigonometric (arctan/tan) transformations:

- For doubly-infinite domains: uses `u = tan(πt/2)` mapping `[-1, 1] → (-∞, ∞)`
- For semi-infinite domains: uses similar tan-based transformations

This can provide better accuracy than the default for integrands that decay like `1/(1+x²)`, such as Lorentzian distributions.

### Alternative: `transformation_cot_inf`

Uses cotangent-based transformations for semi-infinite domains:

- For doubly-infinite domains: falls back to tan transformation
- For semi-infinite `[a, ∞)`: uses a cot-based transformation
- For semi-infinite `(-∞, b]`: uses a cot-based transformation

This can be useful for integrands with specific singularity behavior at finite endpoints or oscillatory integrands.

## Using Custom Transformations

To use an alternative transformation, wrap your algorithm with `ChangeOfVariables`:

```julia
alg = ChangeOfVariables(transformation_function, base_algorithm)
```

where `transformation_function` is one of `transformation_if_inf`, `transformation_tan_inf`, or `transformation_cot_inf`, and `base_algorithm` is the underlying quadrature method like `QuadGKJL()` or `HCubatureJL()`.

## Comparison Example

Let's compare the different transformations on a Gaussian integral. Note that for this
particular integrand, all transformations perform well, but the example demonstrates
the syntax for using alternative transformations.

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
```

You can also explicitly specify which transformation to use with `ChangeOfVariables`:

```julia
# Explicit default transformation
alg_default = ChangeOfVariables(transformation_if_inf, QuadGKJL())
sol_explicit = solve(prob, alg_default)

# Tan transformation
alg_tan = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
sol_tan = solve(prob, alg_tan)

# Cot transformation
alg_cot = ChangeOfVariables(transformation_cot_inf, QuadGKJL())
sol_cot = solve(prob, alg_cot)
```

## Semi-Infinite Integrals

For semi-infinite integrals, the choice of transformation can be more important.
Here's an example integrating `exp(-x)` from 0 to ∞ (true value = 1):

```@example transforms
# Integrating exp(-x) from 0 to ∞ (true value = 1)
f(x, p) = exp(-x)
prob = IntegralProblem(f, (0.0, Inf))

sol_default = solve(prob, QuadGKJL())
println("Default: ", sol_default.u)
```

Alternative transformations can be specified similarly:

```julia
alg_tan = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
sol_tan = solve(prob, alg_tan)

alg_cot = ChangeOfVariables(transformation_cot_inf, QuadGKJL())
sol_cot = solve(prob, alg_cot)
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
