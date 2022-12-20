# [Integral Solver Algorithms](@id solvers)

The following algorithms are available:

- `QuadGKJL`: Uses QuadGK.jl. Requires `nout=1` and `batch=0`.
- `HCubatureJL`: Uses HCubature.jl. Requires `batch=0`.
- `VEGAS`: Uses MonteCarloIntegration.jl. Requires `nout=1`.
- `CubatureJLh`: h-Cubature from Cubature.jl. Requires `using IntegralsCubature`.
- `CubatureJLp`: p-Cubature from Cubature.jl. Requires `using IntegralsCubature`.
- `CubaVegas`: Vegas from Cuba.jl. Requires `using IntegralsCuba`.
- `CubaSUAVE`: SUAVE from Cuba.jl. Requires `using IntegralsCuba`.
- `CubaDivonne`: Divonne from Cuba.jl. Requires `using IntegralsCuba`.
- `CubaCuhre`: Cuhre from Cuba.jl. Requires `using IntegralsCuba`.