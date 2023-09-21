struct TrapezoidalUniformWeights{T} <: UniformWeights
    n::Int
    h::T
end

@inline Base.getindex(w::TrapezoidalUniformWeights, i) = ifelse((i == 1) || (i == w.n), w.h/2 , w.h)


struct TrapezoidalNonuniformWeights{X<:AbstractArray} <: NonuniformWeights
    x::X
end

@inline function Base.getindex(w::TrapezoidalNonuniformWeights, i)
    x = w.x
    (i == firstindex(x)) && return (x[i + 1] - x[i])/2
    (i == lastindex(x)) && return (x[i] - x[i - 1])/2
    return (x[i + 1] - x[i - 1])/2
end

function find_weights(x::AbstractVector, ::TrapezoidalRule)
    x isa AbstractRange && return TrapezoidalUniformWeights(length(x), step(x))
    return TrapezoidalNonuniformWeights(x)
end
