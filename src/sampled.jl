abstract type AbstractWeights end

# must have field `n` for length, and a field `h` for stepsize
abstract type UniformWeights <: AbstractWeights end
@inline Base.iterate(w::UniformWeights) = (0 == w.n) ? nothing : (w[1], 1)
@inline Base.iterate(w::UniformWeights, i) = (i == w.n) ? nothing : (w[i+1], i+1)
Base.length(w::UniformWeights) = w.n
Base.eltype(w::UniformWeights) = typeof(w.h)
Base.size(w::UniformWeights) = (length(w), )

# must contain field `x` which are the sampling points
abstract type NonuniformWeights <: AbstractWeights end 
@inline Base.iterate(w::NonuniformWeights) = (0 == length(w.x)) ? nothing : (w[firstindex(w.x)], firstindex(w.x))
@inline Base.iterate(w::NonuniformWeights, i) = (i == lastindex(w.x)) ? nothing : (w[i+1], i+1)
Base.length(w::NonuniformWeights) = length(w.x)
Base.eltype(w::NonuniformWeights) = eltype(w.x)
Base.size(w::NonuniformWeights) = (length(w), )

_eachslice(data::AbstractArray; dims=ndims(data)) = eachslice(data; dims=dims)
_eachslice(data::AbstractArray{T, 1}; dims=ndims(data)) where T = data


# these can be removed when the Val(dim) is removed from SciMLBase
dimension(::Val{D}) where {D} = D
dimension(D::Int) = D 


function evalrule(data::AbstractArray, weights, dim)
    f = _eachslice(data, dims=dim)
    firstidx, lastidx = firstindex(f), lastindex(f)
    out = f[firstidx]*weights[firstidx]
    if isbits(out)
        for i in firstidx+1:lastidx
            @inbounds out += f[i]*weights[i]
        end
    else
        for i in firstidx+1:lastidx
            @inbounds out .+= f[i] .* weights[i]
        end
    end
    return out

end


# can be reused for other sampled rules
function __solvebp_call(prob::SampledIntegralProblem, alg::TrapezoidalRule; kwargs...)
    dim = dimension(prob.dim)
    err = nothing
    data = prob.y
    grid = prob.x
    weights = find_weights(grid, alg)
    I = evalrule(data, weights, dim)
    return SciMLBase.build_solution(prob, alg, I, err, retcode = ReturnCode.Success)
end


