"""
    AbstractIntegralMetaAlgorithm <: SciMLBase.AbstractIntegralAlgorithm

Abstract type for meta-algorithms that wrap other integration algorithms,
typically to apply transformations or preprocessing steps.
"""
abstract type AbstractIntegralMetaAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end

"""
    ChangeOfVariables(fu2gv, alg)

Apply a change of variables from `∫ f(u,p) du` to an equivalent integral `∫ g(v,p) dv` using
a helper function `fu2gv(f, u_domain) -> (g, v_domain)` where `f` and `g` should be
integral functions. Acts as a wrapper to algorithm `alg`.

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
