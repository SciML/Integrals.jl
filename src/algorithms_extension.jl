## Extension Algorithms

"""
    AbstractIntegralExtensionAlgorithm <: SciMLBase.AbstractIntegralAlgorithm

Abstract type for integration algorithms provided through package extensions.

## Interface

Concrete subtypes are lightweight algorithm configuration objects. The package extension
that owns the backend must implement `Integrals.__solvebp_call(cache, alg, sensealg,
domain, p; kwargs...)` or another solver layer used by `solve!`. Constructors may check
that the required extension is loaded and should store all backend options in fields.
"""
abstract type AbstractIntegralExtensionAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end
"""
    AbstractIntegralCExtensionAlgorithm <: AbstractIntegralExtensionAlgorithm

Abstract type for integration algorithms that use C or C++ libraries through package extensions.

## Interface

Subtypes follow [`AbstractIntegralExtensionAlgorithm`](@ref). Backend wrappers are
responsible for converting Julia integrands, domains, tolerances, and iteration limits to
the C-compatible callback and option formats required by the wrapped library.
"""
abstract type AbstractIntegralCExtensionAlgorithm <: AbstractIntegralExtensionAlgorithm end

"""
    AbstractCubaAlgorithm <: AbstractIntegralCExtensionAlgorithm

Abstract type for integration algorithms from the Cuba.jl package.

## Interface

Concrete Cuba algorithms store Cuba option fields and require `using Cuba` before
construction. Their extension methods must accept multidimensional `IntegralProblem`s,
forward common `solve` tolerances and iteration limits, and return a SciMLBase integral
solution with the backend's estimate and residual/error estimate.
"""
abstract type AbstractCubaAlgorithm <: AbstractIntegralCExtensionAlgorithm end

"""
    CubaVegas(; flags = 0, seed = 0, minevals = 0, nstart = 1000,
        nincrease = 500, gridno = 0)

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Importance sampling is used to reduce variance.

## Keyword Arguments

  - `flags`: Cuba flags bitmask.
  - `seed`: Random seed passed to Cuba.
  - `minevals`: Minimum number of integrand evaluations.
  - `nstart`: Number of evaluations in the first iteration.
  - `nincrease`: Increase in evaluations for later iterations.
  - `gridno`: Cuba grid slot used for state reuse.

## Fields

The fields match the keyword arguments: `flags`, `seed`, `minevals`, `nstart`,
`nincrease`, and `gridno`.

## Returns

Returns a `CubaVegas` algorithm object. `Cuba.jl` must be loaded before construction.

## Example

```julia
using Integrals, Cuba

prob = IntegralProblem((x, p) -> x[1] * x[2], (zeros(2), ones(2)))
sol = solve(prob, CubaVegas(nstart = 2_000))
```

## References

```tex
@article{lepage1978new,
title={A new algorithm for adaptive multidimensional integration},
author={Lepage, G Peter},
journal={Journal of Computational Physics},
volume={27},
number={2},
pages={192--203},
year={1978},
publisher={Elsevier}
}
```
"""
struct CubaVegas <: AbstractCubaAlgorithm
    flags::Int
    seed::Int
    minevals::Int
    nstart::Int
    nincrease::Int
    gridno::Int
end

"""
    CubaSUAVE(; flags = 0, seed = 0, minevals = 0, nnew = 1000,
        nmin = 2, flatness = 25.0)

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Suave stands for subregion-adaptive VEGAS.
Importance sampling and subdivision are thus used to reduce variance.

## Keyword Arguments

  - `flags`: Cuba flags bitmask.
  - `seed`: Random seed passed to Cuba.
  - `minevals`: Minimum number of integrand evaluations.
  - `nnew`: Number of new samples per subdivision.
  - `nmin`: Minimum samples required before subdivision.
  - `flatness`: Flatness parameter used by SUAVE subdivision.

## Fields

The fields match the keyword arguments: `flags`, `seed`, `minevals`, `nnew`, `nmin`,
and `flatness`.

## Returns

Returns a `CubaSUAVE` algorithm object. `Cuba.jl` must be loaded before construction.

## References

```tex
@article{hahn2005cuba,
title={Cuba—a library for multidimensional numerical integration},
author={Hahn, Thomas},
journal={Computer Physics Communications},
volume={168},
number={2},
pages={78--95},
year={2005},
publisher={Elsevier}
}
```
"""
struct CubaSUAVE{R} <: AbstractCubaAlgorithm where {R <: Real}
    flags::Int
    seed::Int
    minevals::Int
    nnew::Int
    nmin::Int
    flatness::R
end

"""
    CubaDivonne(; flags = 0, seed = 0, minevals = 0, key1 = 47, key2 = 1,
        key3 = 1, maxpass = 5, border = 0.0, maxchisq = 10.0,
        mindeviation = 0.25, xgiven = zeros(Cdouble, 0, 0), nextra = 0,
        peakfinder = C_NULL)

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Stratified sampling is used to reduce variance.

## Keyword Arguments

  - `flags`: Cuba flags bitmask.
  - `seed`: Random seed passed to Cuba.
  - `minevals`: Minimum number of integrand evaluations.
  - `key1`, `key2`, `key3`: Divonne rule-selection keys.
  - `maxpass`: Maximum number of partitioning passes.
  - `border`: Border width excluded from partitioning.
  - `maxchisq`: Maximum chi-square value used for consistency checks.
  - `mindeviation`: Minimum relative deviation used for subdivision.
  - `xgiven`: Matrix of user-supplied points.
  - `nextra`: Number of extra points supplied through `peakfinder`.
  - `peakfinder`: Cuba peak-finder callback pointer.

## Fields

The fields match the keyword arguments.

## Returns

Returns a `CubaDivonne` algorithm object. `Cuba.jl` must be loaded before construction.

## References

```tex
@article{friedman1981nested,
title={A nested partitioning procedure for numerical multiple integration},
author={Friedman, Jerome H and Wright, Margaret H},
journal={ACM Transactions on Mathematical Software (TOMS)},
volume={7},
number={1},
pages={76--92},
year={1981},
publisher={ACM New York, NY, USA}
}
```
"""
struct CubaDivonne{R1, R2, R3, R4} <:
    AbstractCubaAlgorithm where {R1 <: Real, R2 <: Real, R3 <: Real, R4 <: Real}
    flags::Int
    seed::Int
    minevals::Int
    key1::Int
    key2::Int
    key3::Int
    maxpass::Int
    border::R1
    maxchisq::R2
    mindeviation::R3
    xgiven::Matrix{R4}
    nextra::Int
    peakfinder::Ptr{Cvoid}
end

"""
    CubaCuhre(; flags = 0, minevals = 0, key = 0)

Multidimensional h-adaptive integration from Cuba.jl.

## Keyword Arguments

  - `flags`: Cuba flags bitmask.
  - `minevals`: Minimum number of integrand evaluations.
  - `key`: Cuba Cuhre rule-selection key.

## Fields

  - `flags::Int`: Stored flags bitmask.
  - `minevals::Int`: Stored minimum evaluation count.
  - `key::Int`: Stored rule-selection key.

## Returns

Returns a `CubaCuhre` algorithm object. `Cuba.jl` must be loaded before construction.

## References

```tex
@article{berntsen1991adaptive,
title={An adaptive algorithm for the approximate calculation of multiple integrals},
author={Berntsen, Jarle and Espelid, Terje O and Genz, Alan},
journal={ACM Transactions on Mathematical Software (TOMS)},
volume={17},
number={4},
pages={437--451},
year={1991},
publisher={ACM New York, NY, USA}
}
```
"""
struct CubaCuhre <: AbstractCubaAlgorithm
    flags::Int
    minevals::Int
    key::Int
end

function CubaVegas(;
        flags = 0, seed = 0, minevals = 0, nstart = 1000, nincrease = 500,
        gridno = 0
    )
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaVegas requires `using Cuba`")
    return CubaVegas(flags, seed, minevals, nstart, nincrease, gridno)
end

function CubaSUAVE(;
        flags = 0, seed = 0, minevals = 0, nnew = 1000, nmin = 2,
        flatness = 25.0
    )
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaSUAVE requires `using Cuba`")
    return CubaSUAVE(flags, seed, minevals, nnew, nmin, flatness)
end

function CubaDivonne(;
        flags = 0, seed = 0, minevals = 0,
        key1 = 47, key2 = 1, key3 = 1, maxpass = 5, border = 0.0,
        maxchisq = 10.0, mindeviation = 0.25,
        xgiven = zeros(Cdouble, 0, 0),
        nextra = 0, peakfinder = C_NULL
    )
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaDivonne requires `using Cuba`")
    return CubaDivonne(
        flags, seed, minevals, key1, key2, key3, maxpass, border, maxchisq,
        mindeviation, xgiven, nextra, peakfinder
    )
end

function CubaCuhre(; flags = 0, minevals = 0, key = 0)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaCuhre requires `using Cuba`")
    return CubaCuhre(flags, minevals, key)
end

"""
    AbstractCubatureJLAlgorithm <: AbstractIntegralCExtensionAlgorithm

Abstract type for integration algorithms from the Cubature.jl package.

## Interface

Subtypes require `using Cubature` before construction and store Cubature.jl option
values. Extension methods must pass common `solve` tolerances and iteration limits to
Cubature.jl and return a SciMLBase integral solution.
"""
abstract type AbstractCubatureJLAlgorithm <: AbstractIntegralCExtensionAlgorithm end

"""
    CubatureJLh(; error_norm = Cubature.INDIVIDUAL)

Multidimensional h-adaptive integration from Cubature.jl.
`error_norm` specifies the convergence criterion  for vector valued integrands.
Defaults to `Cubature.INDIVIDUAL`, other options are
`Cubature.PAIRED`, `Cubature.L1`, `Cubature.L2`, or `Cubature.LINF`.

## Keyword Arguments

  - `error_norm`: Cubature.jl error norm used for vector-valued integrands. The
    constructor default `0` corresponds to `Cubature.INDIVIDUAL`.

## Fields

  - `error_norm::Int32`: Stored Cubature.jl error-norm code.

## Returns

Returns a `CubatureJLh` algorithm object. `Cubature.jl` must be loaded before
construction.

## References

```tex
@article{genz1980remarks,
title={Remarks on algorithm 006: An adaptive algorithm for numerical integration over an N-dimensional rectangular region},
author={Genz, Alan C and Malik, Aftab Ahmad},
journal={Journal of Computational and Applied mathematics},
volume={6},
number={4},
pages={295--302},
year={1980},
publisher={Elsevier}
}
```
"""
struct CubatureJLh <: AbstractCubatureJLAlgorithm
    error_norm::Int32
end
function CubatureJLh(; error_norm = 0)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubatureExt)) &&
        error("CubatureJLh requires `using Cubature`")
    return CubatureJLh(error_norm)
end

"""
    CubatureJLp(; error_norm = Cubature.INDIVIDUAL)

Multidimensional p-adaptive integration from Cubature.jl.
This method is based on repeatedly doubling the degree of the cubature rules,
until convergence is achieved.
The used cubature rule is a tensor product of Clenshaw–Curtis quadrature rules.
`error_norm` specifies the convergence criterion  for vector valued integrands.
Defaults to `Cubature.INDIVIDUAL`, other options are
`Cubature.PAIRED`, `Cubature.L1`, `Cubature.L2`, or `Cubature.LINF`.

## Keyword Arguments

  - `error_norm`: Cubature.jl error norm used for vector-valued integrands. The
    constructor default `0` corresponds to `Cubature.INDIVIDUAL`.

## Fields

  - `error_norm::Int32`: Stored Cubature.jl error-norm code.

## Returns

Returns a `CubatureJLp` algorithm object. `Cubature.jl` must be loaded before
construction.
"""
struct CubatureJLp <: AbstractCubatureJLAlgorithm
    error_norm::Int32
end
function CubatureJLp(; error_norm = 0)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubatureExt)) &&
        error("CubatureJLp requires `using Cubature`")
    return CubatureJLp(error_norm)
end

"""
    ArblibJL(; check_analytic=false, take_prec=false, warn_on_no_convergence=false, opts=C_NULL)

One-dimensional adaptive Gauss-Legendre integration using rigorous error bounds and
precision ball arithmetic. Generally this assumes the integrand is holomorphic or
meromorphic, which is the user's responsibility to verify. The result of the integral is not
guaranteed to satisfy the requested tolerances, however the result is guaranteed to be
within the error estimate.

[Arblib.jl](https://github.com/kalmarek/Arblib.jl) only supports integration of univariate
real- and complex-valued functions with both inplace and out-of-place forms. See their
documentation for additional details the algorithm arguments and on implementing
high-precision integrands. Additionally, the error estimate is included in the return value
of the integral, representing a ball.

## Keyword Arguments

  - `check_analytic`: Whether Arblib should check analyticity assumptions.
  - `take_prec`: Whether to pass precision information through to the integrand.
  - `warn_on_no_convergence`: Whether to warn when Arblib reports non-convergence.
  - `opts`: Arblib option pointer or object passed to the backend.

## Fields

The fields match the keyword arguments: `check_analytic`, `take_prec`,
`warn_on_no_convergence`, and `opts`.

## Returns

Returns an `ArblibJL` algorithm object. `Arblib.jl` must be loaded before construction.
"""
struct ArblibJL{O} <: AbstractIntegralCExtensionAlgorithm
    check_analytic::Bool
    take_prec::Bool
    warn_on_no_convergence::Bool
    opts::O
end
function ArblibJL(;
        check_analytic = false, take_prec = false,
        warn_on_no_convergence = false, opts = C_NULL
    )
    isnothing(Base.get_extension(@__MODULE__, :IntegralsArblibExt)) &&
        error("ArblibJL requires `using Arblib`")
    return ArblibJL(check_analytic, take_prec, warn_on_no_convergence, opts)
end

"""
    VEGASMC(; kws...)

Markov-chain based Vegas algorithm from MCIntegration.jl.

Refer to
[`MCIntegration.integrate`](https://numericaleft.github.io/MCIntegration.jl/dev/lib/montecarlo/#MCIntegration.integrate-Tuple%7BFunction%7D)
for documentation on the keywords, which are passed directly to the solver with a set of
defaults that works for conforming integrands.

## Keyword Arguments

  - `kws...`: Backend keyword arguments forwarded to `MCIntegration.integrate`.

## Fields

  - `kws::NamedTuple`: Stored backend keyword arguments.

## Returns

Returns a `VEGASMC` algorithm object. `MCIntegration.jl` must be loaded before
construction.
"""
struct VEGASMC{K <: NamedTuple} <: AbstractIntegralExtensionAlgorithm
    kws::K
end
function VEGASMC(; kws...)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsMCIntegrationExt)) &&
        error("VEGASMC requires `using MCIntegration`")
    return VEGASMC(NamedTuple(kws))
end

"""
    HAdaptiveIntegrationJL(; kws...)

Adaptive integration over simplices and orthotopes from HAdaptiveIntegration.jl.

This algorithm supports integration over:
- Orthotope domains specified as `(lb, ub)` tuples (converted to `Rectangle`, `Cuboid`, etc.)
- Simplex domains specified directly as `HAdaptiveIntegration.Triangle`,
  `HAdaptiveIntegration.Tetrahedron`, etc.

Any keyword arguments are passed directly to `HAdaptiveIntegration.integrate`.

## Keyword Arguments

  - `kws...`: Backend keyword arguments forwarded to `HAdaptiveIntegration.integrate`.

## Fields

  - `kws::NamedTuple`: Stored backend keyword arguments.

## Returns

Returns an `HAdaptiveIntegrationJL` algorithm object. `HAdaptiveIntegration.jl` must be
loaded before construction.

## Example

```julia
using Integrals, HAdaptiveIntegration

# Orthotope domain (standard lb, ub)
prob = IntegralProblem((x, p) -> x[1] + x[2], (zeros(2), ones(2)))
sol = solve(prob, HAdaptiveIntegrationJL())

# Simplex domain (triangle)
prob = IntegralProblem((x, p) -> x[1] + x[2], Triangle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
sol = solve(prob, HAdaptiveIntegrationJL())
```
"""
struct HAdaptiveIntegrationJL{K <: NamedTuple} <: AbstractIntegralExtensionAlgorithm
    kws::K
end
function HAdaptiveIntegrationJL(; kws...)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsHAdaptiveIntegrationExt)) &&
        error("HAdaptiveIntegrationJL requires `using HAdaptiveIntegration`")
    return HAdaptiveIntegrationJL(NamedTuple(kws))
end

"""
    FastTanhSinhQuadratureJL(; rtol = 1e-12, atol = 0.0, max_levels = 10)

One-dimensional adaptive Tanh-Sinh (double exponential) quadrature from FastTanhSinhQuadrature.jl.
This method uses a double exponential transformation that provides excellent convergence properties,
especially for integrands with endpoint singularities or infinite derivatives at endpoints.

## Keyword Arguments

  - `rtol`: Relative tolerance for convergence (default: `1e-12`)
  - `atol`: Absolute tolerance for convergence (default: `0.0`)
  - `max_levels`: Maximum number of refinement levels in adaptive integration (default: `10`)

## Fields

  - `rtol`: Stored relative tolerance.
  - `atol`: Stored absolute tolerance.
  - `max_levels::Int`: Stored maximum number of refinement levels.

## Returns

Returns a `FastTanhSinhQuadratureJL` algorithm object. FastTanhSinhQuadrature.jl must be
loaded before construction.

## Example

```julia
using Integrals, FastTanhSinhQuadrature

prob = IntegralProblem((x, p) -> sqrt(x), (0.0, 1.0))
sol = solve(prob, FastTanhSinhQuadratureJL(rtol = 1e-10))
```

## Limitations

  - Only supports 1D, 2D, and 3D integration
  - Does not support batched evaluation
  - Does not support in-place integrands

## References

```tex
@article{takahasi1974double,
title={Double exponential formulas for numerical integration},
author={Takahasi, Hidetosi and Mori, Masatake},
journal={Publications of the Research Institute for Mathematical Sciences},
volume={9},
number={3},
pages={721--741},
year={1974},
publisher={Research Institute for Mathematical Sciences}
}
```
"""
struct FastTanhSinhQuadratureJL{T, S} <: AbstractIntegralExtensionAlgorithm
    rtol::T
    atol::S
    max_levels::Int
end
function FastTanhSinhQuadratureJL(; rtol = 1.0e-12, atol = 0.0, max_levels = 10)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsFastTanhSinhQuadratureExt)) &&
        error("FastTanhSinhQuadratureJL requires `using FastTanhSinhQuadrature`")
    return FastTanhSinhQuadratureJL(rtol, atol, max_levels)
end
