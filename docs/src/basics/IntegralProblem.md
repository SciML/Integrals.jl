# [Integral Problems](@id prob)

`IntegralProblem` is a SciMLBase problem type reexported by Integrals.jl for
convenience. Its construction API is documented by SciMLBase; this page describes the
integrand shapes accepted by Integrals.jl algorithms.

The correct shape of the variables (inputs) `u` and the values (outputs) `y` of the integrand `f`
depends on whether batching is used. Batching is enabled by using
[`BatchIntegralFunction`](@ref func) instead of [`IntegralFunction`](@ref func).

**Without batching (using `IntegralFunction`)**

|                       | single variable `f`              | multiple variable `f`            |
|:--------------------- |:-------------------------------- |:-------------------------------- |
| **scalar valued `f`** | `u` is a scalar, `y` is a scalar | `u` is a vector, `y` is a scalar |
| **vector valued `f`** | `u` is a scalar, `y` is a vector | `u` is a vector, `y` is a vector |

**With batching (using `BatchIntegralFunction`)**

|                       | single variable `f`              | multiple variable `f`            |
|:--------------------- |:-------------------------------- |:-------------------------------- |
| **scalar valued `f`** | `u` is a vector, `y` is a vector | `u` is a matrix, `y` is a vector |
| **vector valued `f`** | `u` is a vector, `y` is a matrix | `u` is a matrix, `y` is a matrix |

The last dimension is always used as the batching dimension,
e.g., if `u` is a matrix, then `u[:,i]` is the `i`th point where the integrand will be evaluated.
