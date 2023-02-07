"""
    QuadGKJL(; order = 7, norm=norm)

One-dimensional Gauss-Kronrod integration from QuadGK.jl.
This method also takes the optional arguments `order` and `norm`.
Which are the order of the integration rule
and the norm for calculating the error, respectively

## References

@article{laurie1997calculation,
title={Calculation of Gauss-Kronrod quadrature rules},
author={Laurie, Dirk},
journal={Mathematics of Computation},
volume={66},
number={219},
pages={1133--1145},
year={1997}
}
"""
struct QuadGKJL{F} <: SciMLBase.AbstractIntegralAlgorithm where {F}
    order::Int
    norm::F
end
QuadGKJL(; order = 7, norm = norm) = QuadGKJL(order, norm)

"""
    HCubatureJL(; norm=norm, initdiv=1)

Multidimensional "h-adaptive" integration from HCubature.jl.
This method also takes the optional arguments `initdiv` and `norm`.
Which are the intial number of segments
each dimension of the integration domain is divided into,
and the norm for calculating the error, respectively.

## References

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
"""
struct HCubatureJL{F} <: SciMLBase.AbstractIntegralAlgorithm where {F}
    initdiv::Int
    norm::F
end
HCubatureJL(; initdiv = 1, norm = norm) = HCubatureJL(initdiv, norm)

"""
    VEGAS(; nbins = 100, ncalls = 1000, debug=false)

Multidimensional adaptive Monte Carlo integration from MonteCarloIntegration.jl.
Importance sampling is used to reduce variance.
This method also takes three optional arguments `nbins`, `ncalls` and `debug`
which are the intial number of bins
each dimension of the integration domain is divided into,
the number of function calls per iteration of the algorithm,
and whether debug info should be printed, respectively.

## References

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
"""
struct VEGAS <: SciMLBase.AbstractIntegralAlgorithm
    nbins::Int
    ncalls::Int
    debug::Bool
end
VEGAS(; nbins = 100, ncalls = 1000, debug = false) = VEGAS(nbins, ncalls, debug)
