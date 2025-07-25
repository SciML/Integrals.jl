"""
    AbstractWeights

Abstract supertype for all quadrature weight types used in sampled integration methods.
"""
abstract type AbstractWeights end

"""
    UniformWeights <: AbstractWeights

Abstract type for quadrature weights with uniform (equally-spaced) sampling points.

Implementations must have:
- field `n` for the number of points
- field `h` for the step size between points
"""
abstract type UniformWeights <: AbstractWeights end
@inline Base.iterate(w::UniformWeights) = (0 == w.n) ? nothing : (w[1], 1)
@inline Base.iterate(w::UniformWeights, i) = (i == w.n) ? nothing : (w[i + 1], i + 1)
Base.length(w::UniformWeights) = w.n
Base.eltype(w::UniformWeights) = typeof(w.h)
Base.size(w::UniformWeights) = (length(w),)

"""
    NonuniformWeights <: AbstractWeights

Abstract type for quadrature weights with non-uniform (arbitrarily-spaced) sampling points.

Implementations must have:
- field `x` containing the sampling points
"""
abstract type NonuniformWeights <: AbstractWeights end
@inline Base.iterate(w::NonuniformWeights) = (0 == length(w.x)) ? nothing :
                                             (w[firstindex(w.x)], firstindex(w.x))
@inline Base.iterate(w::NonuniformWeights, i) = (i == lastindex(w.x)) ? nothing :
                                                (w[i + 1], i + 1)
Base.length(w::NonuniformWeights) = length(w.x)
Base.eltype(w::NonuniformWeights) = eltype(w.x)
Base.size(w::NonuniformWeights) = (length(w),)

_eachslice(data::AbstractArray; dims = ndims(data)) = eachslice(data; dims = dims)
_eachslice(data::AbstractArray{T, 1}; dims = ndims(data)) where {T} = data

# these can be removed when the Val(dim) is removed from SciMLBase
dimension(::Val{D}) where {D} = D
dimension(D::Int) = D

function evalrule(data::AbstractArray, weights, dim)
    fw = zip(_eachslice(data, dims = dim), weights)
    next = iterate(fw)
    next === nothing && throw(ArgumentError("No points to integrate"))
    (f1, w1), state = next
    out = w1 * f1
    next = iterate(fw, state)
    if isbits(out)
        while next !== nothing
            (fi, wi), state = next
            out += wi * fi
            next = iterate(fw, state)
        end
    else
        while next !== nothing
            (fi, wi), state = next
            out .+= wi .* fi
            next = iterate(fw, state)
        end
    end
    return out
end

# can be reused for other sampled rules, which should implement find_weights(x, alg)

function init_cacheval(alg::SciMLBase.AbstractIntegralAlgorithm,
        prob::SampledIntegralProblem)
    find_weights(prob.x, alg)
end

function __solvebp_call(cache::SampledIntegralCache,
        alg::SciMLBase.AbstractIntegralAlgorithm;
        kwargs...)
    dim = dimension(cache.dim)
    err = nothing
    data = cache.y
    grid = cache.x
    if cache.isfresh
        cache.cacheval = find_weights(grid, alg)
        cache.isfresh = false
    end
    weights = cache.cacheval
    I = evalrule(data, weights, dim)
    prob = build_problem(cache)
    return SciMLBase.build_solution(prob, alg, I, err, retcode = ReturnCode.Success)
end
