struct TrapezoidalUniformWeights <: UniformWeights
    n::Int
    h::Float64
end

@inline Base.getindex(w::TrapezoidalUniformWeights, i) = ifelse((i == 1) || (i == w.n), w.h*0.5 , w.h)


struct TrapezoidalNonuniformWeights{X<:AbstractArray} <: NonuniformWeights
    x::X
end

@inline function Base.getindex(w::TrapezoidalNonuniformWeights, i)
    x = w.x
    (i == firstindex(x)) && return (x[i + 1] - x[i])*0.5
    (i == lastindex(x)) && return (x[i] - x[i - 1])*0.5
    return (x[i + 1] - x[i - 1])*0.5
end

function find_weights(x::AbstractVector, ::TrapezoidalRule)
    x isa AbstractRange && return TrapezoidalUniformWeights(length(x), step(x))
    return TrapezoidalNonuniformWeights(x)
end
