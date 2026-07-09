"""
    QuadGKJL(; order = 7, norm = norm, buffer = nothing)

One-dimensional adaptive Gauss-Kronrod integration from QuadGK.jl.

## Keyword Arguments

  - `order`: Order of the embedded Gauss-Kronrod rule. Higher values reduce the number
    of subintervals for smooth integrands and increase work per subinterval.
  - `norm`: Function used by QuadGK.jl to turn the integral estimate and error estimate
    into scalar convergence criteria. This is most useful for array-valued integrands.
  - `buffer`: If non-`nothing`, allocate and cache a QuadGK segment buffer during
    `init`. This avoids repeated buffer allocation when `solve!` is called on the same
    cache. Buffer construction may evaluate the integrand unless the problem supplies an
    `integrand_prototype`.

## Fields

  - `order::Int`: Stored quadrature rule order.
  - `norm`: Stored convergence norm.
  - `buffer`: Stored buffer option.

## Returns

Returns a `QuadGKJL` algorithm object for use with
`solve(prob::IntegralProblem, alg)`, `init`, and `solve!`.

## Example

```julia
using Integrals

prob = IntegralProblem((x, p) -> x^2, (0.0, 1.0))
sol = solve(prob, QuadGKJL(order = 9); reltol = 1e-10)
```

## References

```tex
@article{laurie1997calculation,
title={Calculation of Gauss-Kronrod quadrature rules},
author={Laurie, Dirk},
journal={Mathematics of Computation},
volume={66},
number={219},
pages={1133--1145},
year={1997}
}
```
"""
struct QuadGKJL{F, B} <: SciMLBase.AbstractIntegralAlgorithm
    order::Int
    norm::F
    buffer::B
end
QuadGKJL(; order = 7, norm = norm, buffer = nothing) = QuadGKJL(order, norm, buffer)

"""
    HCubatureJL(; norm=norm, initdiv=1, buffer = nothing)

Multidimensional h-adaptive integration from HCubature.jl.

## Keyword Arguments

  - `initdiv`: Initial number of segments used to divide each integration dimension.
  - `norm`: Function used by HCubature.jl to turn integral and error estimates into
    scalar convergence criteria. This is most useful for array-valued integrands.
  - `buffer`: If non-`nothing`, allocate and cache HCubature work buffers during
    `init`. The buffer is managed by Integrals.jl; callers should not pass an HCubature
    buffer directly.

## Fields

  - `initdiv::Int`: Stored initial subdivision count.
  - `norm`: Stored convergence norm.
  - `buffer`: Stored buffer option.

## Returns

Returns an `HCubatureJL` algorithm object for `IntegralProblem`s over scalar or vector
box domains.

## Example

```julia
using Integrals

f(x, p) = x[1]^2 + x[2]^2
prob = IntegralProblem(f, ([0.0, 0.0], [1.0, 1.0]))
sol = solve(prob, HCubatureJL(initdiv = 2))
```

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
struct HCubatureJL{F, B} <: SciMLBase.AbstractIntegralAlgorithm
    initdiv::Int
    norm::F
    buffer::B
end
function HCubatureJL(; initdiv = 1, norm = norm, buffer = nothing)
    return HCubatureJL(initdiv, norm, buffer)
end

"""
    VEGAS(; nbins = 100, ncalls = 1000, debug=false, seed = nothing)

Multidimensional adaptive Monte Carlo integration from MonteCarloIntegration.jl.
Importance sampling is used to reduce variance.

## Keyword Arguments

  - `nbins`: Initial number of bins used for each integration dimension.
  - `ncalls`: Number of integrand calls requested per Monte Carlo iteration.
  - `debug`: Whether to request debug output from MonteCarloIntegration.jl.
  - `seed`: Optional random seed passed to the underlying algorithm.

## Fields

  - `nbins::Int`: Stored bin count.
  - `ncalls::Int`: Stored calls-per-iteration target.
  - `debug::Bool`: Stored debug-output flag.
  - `seed`: Stored seed value.

## Returns

Returns a `VEGAS` algorithm object for multidimensional scalar Monte Carlo
integration.

## Example

```julia
using Integrals

f(x, p) = exp(-sum(abs2, x))
prob = IntegralProblem(f, (zeros(3), ones(3)))
sol = solve(prob, VEGAS(ncalls = 2_000); reltol = 1e-2)
```

## Limitations

This algorithm can only integrate scalar `Float64`-valued functions.

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
struct VEGAS{S} <: SciMLBase.AbstractIntegralAlgorithm
    nbins::Int
    ncalls::Int
    debug::Bool
    seed::S
end
function VEGAS(; nbins = 100, ncalls = 1000, debug = false, seed = nothing)
    return VEGAS(nbins, ncalls, debug, seed)
end

"""
    GaussLegendre(; n = 250, subintervals = 1, nodes = nothing, weights = nothing)
    GaussLegendre(nodes, weights, subintervals = 1)

Fixed-order Gauss-Legendre quadrature, optionally applied on a composite partition of
the integration interval.

## Arguments

  - `nodes`: Quadrature nodes on the standard interval `[-1, 1]`.
  - `weights`: Quadrature weights corresponding to `nodes`.
  - `subintervals`: Number of equally sized subintervals used for composite
    Gauss-Legendre quadrature.

## Keyword Arguments

  - `n`: Number of quadrature nodes to construct when `nodes` or `weights` is
    `nothing`.
  - `subintervals`: Number of subintervals used for composite quadrature. Must be
    positive.
  - `nodes`: Optional precomputed nodes. If omitted, `gausslegendre(n)` is used.
  - `weights`: Optional precomputed weights. If omitted, `gausslegendre(n)` is used.

## Fields

  - `nodes`: Stored quadrature nodes.
  - `weights`: Stored quadrature weights.
  - `subintervals::Int64`: Stored number of subintervals.

The type parameter `C` is `true` when `subintervals > 1` and `false` otherwise.
Composite mode splits `[a, b]` into `subintervals` pieces and applies the same rule to
each piece.

## Returns

Returns a `GaussLegendre` algorithm object. `GaussLegendre(; n=...)` requires
FastGaussQuadrature.jl to be loaded so that the `gausslegendre` extension method is
available.

## Example

```julia
using Integrals, FastGaussQuadrature

prob = IntegralProblem((x, p) -> cos(x), (0.0, pi / 2))
sol = solve(prob, GaussLegendre(n = 64))
```
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

Evaluate a user-supplied fixed quadrature rule.

The rule function `q` must support `nodes, weights = q(n)` and return nodes and weights
for the standard interval or hypercube `[-1, 1]^d`. Integrals.jl rescales the nodes to
the problem domain before evaluating the integrand. Nodes may be scalars in one
dimension or vectors in multiple dimensions; weights must be scalar.

## Arguments

  - `q`: Function that returns `(nodes, weights)` for a requested node count.

## Keyword Arguments

  - `n`: Number of quadrature nodes requested from `q`. Must be positive.

## Fields

  - `q`: Stored quadrature rule constructor.
  - `n::Int`: Stored quadrature node count.

## Returns

Returns a `QuadratureRule` algorithm object. The method computes the fixed quadrature
sum and reports success; callers are responsible for checking convergence by changing
`n` or otherwise validating the chosen rule.

## Example

```julia
using Integrals, FastGaussQuadrature

prob = IntegralProblem((x, p) -> x^4, (-1.0, 1.0))
sol = solve(prob, QuadratureRule(gausslegendre; n = 8))
```
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
