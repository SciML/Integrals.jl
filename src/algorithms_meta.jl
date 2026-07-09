"""
    AbstractIntegralMetaAlgorithm <: SciMLBase.AbstractIntegralAlgorithm

Abstract type for meta-algorithms that wrap other integration algorithms,
typically to apply transformations or preprocessing steps.

## Interface

Concrete subtypes must wrap an underlying `SciMLBase.AbstractIntegralAlgorithm` and
must implement the same solver lifecycle as ordinary integral algorithms:

  - `init_cacheval(alg, prob)` may allocate reusable cache state.
  - `__solve(cache, alg, sensealg, domain, p; kwargs...)` or the lower
    `__solvebp_call` layer must transform the problem and delegate to the wrapped
    algorithm.
  - returned solutions must be built for the user's original problem when the
    transformation is transparent to callers.
"""
abstract type AbstractIntegralMetaAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

"""
    ChangeOfVariables(fu2gv, alg)

Apply a change of variables from `∫ f(u,p) du` to an equivalent integral `∫ g(v,p) dv` using
a helper function `fu2gv(f, u_domain) -> (g, v_domain)`. The transformed integrand `g`
must obey the same `IntegralFunction` or `BatchIntegralFunction` calling convention as
`f`.

This meta-algorithm allows users to apply custom or alternative transformations when
integrating, particularly useful for handling infinite domains where different
transformations may provide better accuracy for specific integrand types.

## Arguments

- `fu2gv`: A transformation function with signature `(f, domain) -> (g, new_domain)` that transforms
  the integrand and domain. Built-in options include:
  - [`transformation_if_inf`](@ref): Default rational transformation (used automatically)
  - [`transformation_tan_inf`](@ref): Arctan/tan transformation
  - [`transformation_cot_inf`](@ref): Cotangent transformation for semi-infinite domains
- `alg`: The underlying integration algorithm to use (e.g., `QuadGKJL()`, `HCubatureJL()`)

## Fields

  - `fu2gv`: Stored transformation function.
  - `alg`: Stored wrapped algorithm.

## Returns

Returns a `ChangeOfVariables` meta-algorithm. When solved, the result is reported as a
solution of the original integral problem.

## Example

```julia
using Integrals

f(x, p) = exp(-x^2)
prob = IntegralProblem(f, (-Inf, Inf))

# Use alternative tan transformation instead of default
alg = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
sol = solve(prob, alg)
```

See also: [`transformation_if_inf`](@ref), [`transformation_tan_inf`](@ref), [`transformation_cot_inf`](@ref)
"""
struct ChangeOfVariables{T, A <: SciMLBase.AbstractIntegralAlgorithm} <:
    AbstractIntegralMetaAlgorithm
    fu2gv::T
    alg::A
end
