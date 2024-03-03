"""
    QuadGKJL(; order = 7, norm = norm, buffer = nothing)

One-dimensional Gauss-Kronrod integration from QuadGK.jl.
This method also takes the optional arguments `order` and `norm`.
Which are the order of the integration rule
and the norm for calculating the error, respectively.
Lastly, the `buffer` keyword, if set (e.g. `buffer=true`), will allocate a buffer to reuse
for multiple integrals and may require evaluating the integrand unless an
`integrand_prototype` is provided. Unlike the `segbuf` keyword to `quadgk`, you do not
allocate the buffer as this is handled automatically.

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
struct QuadGKJL{F, B} <: SciMLBase.AbstractIntegralAlgorithm
    order::Int
    norm::F
    buffer::B
end
QuadGKJL(; order = 7, norm = norm, buffer = nothing) = QuadGKJL(order, norm, buffer)

"""
    HCubatureJL(; norm=norm, initdiv=1, buffer = nothing)

Multidimensional "h-adaptive" integration from HCubature.jl.
This method also takes the optional arguments `initdiv` and `norm`.
Which are the initial number of segments
each dimension of the integration domain is divided into,
and the norm for calculating the error, respectively.
Lastly, the `buffer` keyword, if set (e.g. `buffer=true`), will allocate a buffer to reuse
for multiple integrals and may require evaluating the integrand unless an
`integrand_prototype` is provided. Unlike the `buffer` keyword to `hcubature/hquadrature`,
you do not allocate the buffer as this is handled automatically.

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
struct HCubatureJL{F, B} <: SciMLBase.AbstractIntegralAlgorithm
    initdiv::Int
    norm::F
    buffer::B
end
function HCubatureJL(; initdiv = 1, norm = norm, buffer = nothing)
    HCubatureJL(initdiv, norm, buffer)
end

"""
    VEGAS(; nbins = 100, ncalls = 1000, debug=false, seed = nothing)

Multidimensional adaptive Monte Carlo integration from MonteCarloIntegration.jl.
Importance sampling is used to reduce variance.
This method also takes three optional arguments `nbins`, `ncalls` and `debug`
which are the initial number of bins
each dimension of the integration domain is divided into,
the number of function calls per iteration of the algorithm,
and whether debug info should be printed, respectively.

## Limitations

This algorithm can only integrate `Float64`-valued functions

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
struct VEGAS{S} <: SciMLBase.AbstractIntegralAlgorithm
    nbins::Int
    ncalls::Int
    debug::Bool
    seed::S
end
function VEGAS(; nbins = 100, ncalls = 1000, debug = false, seed = nothing)
    VEGAS(nbins, ncalls, debug, seed)
end

"""
    GaussLegendre{C, N, W}

Struct for evaluating an integral via (composite) Gauss-Legendre quadrature.
The field `C` will be `true` if `subintervals > 1`, and `false` otherwise.

The fields `nodes::N` and `weights::W` are defined by
`nodes, weights = gausslegendre(n)` for a given number of nodes `n`.

The field `subintervals::Int64 = 1` (with default value `1`) defines the
number of intervals to partition the original interval of integration
`[a, b]` into, splitting it into `[xⱼ, xⱼ₊₁]` for `j = 1,…,subintervals`,
where `xⱼ = a + (j-1)h` and `h = (b-a)/subintervals`. Gauss-Legendre
quadrature is then applied on each subinterval. For example, if
`[a, b] = [-1, 1]` and `subintervals = 2`, then Gauss-Legendre
quadrature will be applied separately on `[-1, 0]` and `[0, 1]`,
summing the two results.
"""
struct GaussLegendre{C, N, W} <: SciMLBase.AbstractIntegralAlgorithm
    nodes::N
    weights::W
    subintervals::Int64
    function GaussLegendre(nodes::N, weights::W, subintervals = 1) where {N, W}
        if subintervals > 1
            return new{true, N, W}(nodes, weights, subintervals)
        elseif subintervals == 1
            return new{false, N, W}(nodes, weights, subintervals)
        else
            throw(ArgumentError("Cannot use a nonpositive number of subintervals."))
        end
    end
end
function gausslegendre end
function GaussLegendre(; n = 250, subintervals = 1, nodes = nothing, weights = nothing)
    if isnothing(nodes) || isnothing(weights)
        nodes, weights = gausslegendre(n)
    end
    return GaussLegendre(nodes, weights, subintervals)
end

"""
QuadratureRule(q; n=250)

Algorithm to construct and evaluate a quadrature rule `q` of `n` points computed from the
inputs as `x, w = q(n)`. It assumes the nodes and weights are for the standard interval
`[-1, 1]^d` in `d` dimensions, and rescales the nodes to the specific hypercube being
solved. The nodes `x` may be scalars in 1d or vectors in arbitrary dimensions, and the
weights `w` must be scalar. The algorithm computes the quadrature rule `sum(w .* f.(x))` and
the caller must check that the result is converged with respect to `n`.
"""
struct QuadratureRule{Q} <: SciMLBase.AbstractIntegralAlgorithm
    q::Q
    n::Int
    function QuadratureRule(q::Q, n::Integer) where {Q}
        n > 0 ||
            throw(ArgumentError("Cannot use a nonpositive number of quadrature nodes."))
        return new{Q}(q, n)
    end
end
QuadratureRule(q; n = 250) = QuadratureRule(q, n)
