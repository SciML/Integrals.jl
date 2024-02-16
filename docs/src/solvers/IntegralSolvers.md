# [Integral Solver Algorithms](@id solvers)

The following algorithms are available:

  - `QuadGKJL`: Uses QuadGK.jl, which supports one-dimensional integration of scalar and array-valued integrands with in-place or batched forms. Integrands that are both in-place and batched are implemented in the wrapper but are not supported under the hood.
  - `HCubatureJL`: Uses HCubature.jl, which supports scalar and array-valued integrands and works best in low dimensions, e.g. ≤ 8. In-place integrands are implemented in the wrapper but are not supported under the hood. Batching is not supported.
  - `VEGAS`: Uses MonteCarloIntegration.jl, which requires scalar, `Float64`-valued integrands and works in any number of dimensions.
  - `VEGASMC`: Uses MCIntegration.jl. Requires `using MCIntegration`. Doesn't support batching.
  - `CubatureJLh`: h-Cubature from Cubature.jl. Requires `using Cubature`.
  - `CubatureJLp`: p-Cubature from Cubature.jl. Requires `using Cubature`.
  - `CubaVegas`: Vegas from Cuba.jl. Requires `using Cuba`.
  - `CubaSUAVE`: SUAVE from Cuba.jl. Requires `using Cuba`.
  - `CubaDivonne`: Divonne from Cuba.jl. Requires `using Cuba`. Works only for `>1`-dimensional integrations.
  - `CubaCuhre`: Cuhre from Cuba.jl. Requires `using Cuba`. Works only for `>1`-dimensional integrations.
  - `GaussLegendre`: Performs fixed-order Gauss-Legendre quadrature. Requires `using FastGaussQuadrature`.
  - `QuadratureRule`: Accepts a user-defined function that returns nodes and weights.
  - `ArblibJL`: real- and complex-valued univariate integration of holomorphic
    and meromorphic functions from Arblib.jl. Requires `using Arblib`.

```@docs
QuadGKJL
HCubatureJL
CubatureJLp
CubatureJLh
VEGAS
VEGASMC
CubaVegas
CubaSUAVE
CubaDivonne
CubaCuhre
GaussLegendre
QuadratureRule
ArblibJL
```
