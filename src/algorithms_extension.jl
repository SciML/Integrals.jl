## Extension Algorithms

abstract type AbstractIntegralExtensionAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end
abstract type AbstractIntegralCExtensionAlgorithm <: AbstractIntegralExtensionAlgorithm end

abstract type AbstractCubaAlgorithm <: AbstractIntegralCExtensionAlgorithm end

"""
    CubaVegas()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Importance sampling is used to reduce variance.

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
    CubaSUAVE()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Suave stands for subregion-adaptive VEGAS.
Importance sampling and subdivision are thus used to reduce variance.

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
    CubaDivonne()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Stratified sampling is used to reduce variance.

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
    CubaCuhre()

Multidimensional h-adaptive integration from Cuba.jl.

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

function CubaVegas(; flags = 0, seed = 0, minevals = 0, nstart = 1000, nincrease = 500,
        gridno = 0)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaVegas requires `using Cuba`")
    return CubaVegas(flags, seed, minevals, nstart, nincrease, gridno)
end

function CubaSUAVE(; flags = 0, seed = 0, minevals = 0, nnew = 1000, nmin = 2,
        flatness = 25.0)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaSUAVE requires `using Cuba`")
    return CubaSUAVE(flags, seed, minevals, nnew, nmin, flatness)
end

function CubaDivonne(; flags = 0, seed = 0, minevals = 0,
        key1 = 47, key2 = 1, key3 = 1, maxpass = 5, border = 0.0,
        maxchisq = 10.0, mindeviation = 0.25,
        xgiven = zeros(Cdouble, 0, 0),
        nextra = 0, peakfinder = C_NULL)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaDivonne requires `using Cuba`")
    return CubaDivonne(flags, seed, minevals, key1, key2, key3, maxpass, border, maxchisq,
        mindeviation, xgiven, nextra, peakfinder)
end

function CubaCuhre(; flags = 0, minevals = 0, key = 0)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsCubaExt)) &&
        error("CubaCuhre requires `using Cuba`")
    return CubaCuhre(flags, minevals, key)
end

abstract type AbstractCubatureJLAlgorithm <: AbstractIntegralCExtensionAlgorithm end

"""
    CubatureJLh(; error_norm=Cubature.INDIVIDUAL)

Multidimensional h-adaptive integration from Cubature.jl.
`error_norm` specifies the convergence criterion  for vector valued integrands.
Defaults to `Cubature.INDIVIDUAL`, other options are
`Cubature.PAIRED`, `Cubature.L1`, `Cubature.L2`, or `Cubature.LINF`.

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
    CubatureJLp(; error_norm=Cubature.INDIVIDUAL)

Multidimensional p-adaptive integration from Cubature.jl.
This method is based on repeatedly doubling the degree of the cubature rules,
until convergence is achieved.
The used cubature rule is a tensor product of Clenshaw–Curtis quadrature rules.
`error_norm` specifies the convergence criterion  for vector valued integrands.
Defaults to `Cubature.INDIVIDUAL`, other options are
`Cubature.PAIRED`, `Cubature.L1`, `Cubature.L2`, or `Cubature.LINF`.
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
"""
struct ArblibJL{O} <: AbstractIntegralCExtensionAlgorithm
    check_analytic::Bool
    take_prec::Bool
    warn_on_no_convergence::Bool
    opts::O
end
function ArblibJL(; check_analytic = false, take_prec = false,
        warn_on_no_convergence = false, opts = C_NULL)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsArblibExt)) &&
        error("ArblibJL requires `using Arblib`")
    return ArblibJL(check_analytic, take_prec, warn_on_no_convergence, opts)
end

"""
    VEGASMC(; kws...)

Markov-chain based Vegas algorithm from MCIntegration.jl

Refer to
[`MCIntegration.integrate`](https://numericaleft.github.io/MCIntegration.jl/dev/lib/montecarlo/#MCIntegration.integrate-Tuple%7BFunction%7D)
for documentation on the keywords, which are passed directly to the solver with a set of
defaults that works for conforming integrands.
"""
struct VEGASMC{K <: NamedTuple} <: AbstractIntegralExtensionAlgorithm
    kws::K
end
function VEGASMC(; kws...)
    isnothing(Base.get_extension(@__MODULE__, :IntegralsMCIntegrationExt)) &&
        error("VEGASMC requires `using MCIntegration`")
    return VEGASMC(NamedTuple(kws))
end
