"""
    TrapezoidalUniformWeights{T} <: UniformWeights

Quadrature weights for the trapezoidal rule with uniformly spaced points.

# Fields

  - `n::Int`: Number of points
  - `h::T`: Step size between points
"""
struct TrapezoidalUniformWeights{T} <: UniformWeights
    n::Int
    h::T
end

@inline Base.getindex(w::TrapezoidalUniformWeights, i) = ifelse((i == 1) || (i == w.n),
    w.h / 2,
    w.h)

"""
    TrapezoidalNonuniformWeights{X <: AbstractArray} <: NonuniformWeights

Quadrature weights for the trapezoidal rule with non-uniformly spaced points.

# Fields

  - `x::X`: Array of sampling points
"""
struct TrapezoidalNonuniformWeights{X <: AbstractArray} <: NonuniformWeights
    x::X
end

@inline function Base.getindex(w::TrapezoidalNonuniformWeights, i)
    x = w.x
    (i == firstindex(x)) && return (x[i + 1] - x[i]) / 2
    (i == lastindex(x)) && return (x[i] - x[i - 1]) / 2
    return (x[i + 1] - x[i - 1]) / 2
end

function find_weights(x::AbstractVector, ::TrapezoidalRule)
    x isa AbstractRange && return TrapezoidalUniformWeights(length(x), step(x))
    return TrapezoidalNonuniformWeights(x)
end
