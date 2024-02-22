struct SimpsonUniformWeights{T} <: UniformWeights
    n::Int
    h::T
end

@inline function Base.getindex(w::SimpsonUniformWeights, i)
    # evenly spaced simpson's 1/3, 3/8 rule 
    h = w.h
    n = w.n
    (i == 1 || i == n) && return 17h / 48
    (i == 2 || i == n - 1) && return 59h / 48
    (i == 3 || i == n - 2) && return 43h / 48
    (i == 4 || i == n - 3) && return 49h / 48
    return h
end

struct SimpsonNonuniformWeights{X <: AbstractArray} <: NonuniformWeights
    x::X
end

@inline function Base.getindex(w::SimpsonNonuniformWeights, i)
    # composite 1/3 rule for irregular grids

    checkbounds(w.x, i)
    x = w.x
    j = i - firstindex(x)
    @assert length(w.x)>2 "The length of the grid must exceed 2 for simpsons rule."
    i == firstindex(x) && return (x[begin + 2] - x[begin + 0]) / 6 *
           (2 - (x[begin + 2] - x[begin + 1]) / (x[begin + 1] - x[begin + 0]))

    if isodd(length(x)) # even number of subintervals
        i == lastindex(x) && return (x[end] - x[end - 2]) / 6 *
               (2 - (x[end - 1] - x[end - 2]) / (x[end] - x[end - 1]))
    else # odd number of subintervals, we add additional terms
        i == lastindex(x) && return (x[end] - x[end - 1]) *
               (2 * (x[end] - x[end - 1]) + 3 * (x[end - 1] - x[end - 2])) /
               (x[end] - x[end - 2]) / 6
        i == lastindex(x) - 1 && return (x[end - 1] - x[end - 3]) / 6 *
               (2 - (x[end - 2] - x[end - 3]) / (x[end - 1] - x[end - 2])) +
               (x[end] - x[end - 1]) *
               ((x[end] - x[end - 1]) + 3 * (x[end - 1] - x[end - 2])) /
               (x[end - 1] - x[end - 2]) / 6
        i == lastindex(x) - 2 &&
            return (x[end - 1] - x[end - 3])^3 / (x[end - 2] - x[end - 3]) /
                   (x[end - 1] - x[end - 2]) / 6 -
                   (x[end] - x[end - 1])^3 / (x[end - 1] - x[end - 2]) /
                   (x[end] - x[end - 2]) / 6
    end
    isodd(j) &&
        return (x[begin + j + 1] - x[begin + j - 1])^3 / (x[begin + j] - x[begin + j - 1]) /
               (x[begin + j + 1] - x[begin + j]) / 6
    iseven(j) &&
        return (x[begin + j] - x[begin + j - 2]) / 6 *
               (2 -
                (x[begin + j - 1] - x[begin + j - 2]) / (x[begin + j] - x[begin + j - 1])) +
               (x[begin + j + 2] - x[begin + j]) / 6 *
               (2 -
                (x[begin + j + 2] - x[begin + j + 1]) / (x[begin + j + 1] - x[begin + j]))
end

function find_weights(x::AbstractVector, ::SimpsonsRule)
    x isa AbstractRange && return SimpsonUniformWeights(length(x), step(x))
    return SimpsonNonuniformWeights(x)
end
