# [Integral Problems](@id prob)

```@docs
SciMLBase.IntegralProblem
```

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
