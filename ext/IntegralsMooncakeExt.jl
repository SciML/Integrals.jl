module IntegralsMooncakeExt
using Mooncake
using LinearAlgebra: dot
using Integrals, SciMLBase, QuadGK
using Mooncake: @from_chainrules, @is_primitive, increment!!, MinimalCtx, NoFData,
    CoDual, primal, NoRData, zero_fcodual, increment_and_get_rdata!, @zero_derivative
using Integrals: AbstractIntegralMetaAlgorithm, IntegralProblem
import ChainRulesCore
import ChainRulesCore: Tangent, NoTangent, ProjectTo
using Zygote # use chainrules defined in ZygoteExt

batch_unwrap(x::AbstractArray) = dropdims(x; dims = ndims(x))

@zero_derivative MinimalCtx Tuple{typeof(QuadGK.quadgk), Vararg}
@zero_derivative MinimalCtx Tuple{typeof(QuadGK.cachedrule), Any, Integer}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.checkkwargs), Vararg}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.isinplace), Vararg}
@zero_derivative MinimalCtx Tuple{
    typeof(Integrals.init_cacheval),
    Union{<:SciMLBase.AbstractIntegralAlgorithm, <:AbstractIntegralMetaAlgorithm},
    Union{<:IntegralProblem, <:SampledIntegralProblem},
}
@zero_derivative MinimalCtx Tuple{
    typeof(Integrals.substitute_f),
    Union{<:BatchIntegralFunction, <:IntegralFunction}, Any, Any, Any,
}
@zero_derivative MinimalCtx Tuple{
    typeof(Integrals.substitute_v), Any, Any,
    Union{<:AbstractVector, <:Number}, Union{<:AbstractVector, <:Number},
}
@zero_derivative MinimalCtx Tuple{
    typeof(Integrals.substitute_bv), Any, AbstractArray,
    Union{<:AbstractVector, <:Number}, Union{<:AbstractVector, <:Number},
}

@from_chainrules MinimalCtx Tuple{Type{IntegralProblem{iip}}, Any, Any, Any} where {iip} true
@from_chainrules MinimalCtx Tuple{Type{IntegralProblem}, Any, Any, Any} true
@from_chainrules MinimalCtx Tuple{typeof(Integrals.u2t), Any, Any} true
@from_chainrules MinimalCtx Tuple{typeof(Integrals.__solvebp), Any, Any, Any, Any, Any} true

# Add MooncakeVJP support to the dispatch function defined in ZygoteExt
function Integrals._compute_dfdp_and_f(::Integrals.MooncakeVJP, cache, p, Δ)
    # Extract the tangent value - Mooncake wraps it in a struct with .u field
    Δ_val = hasproperty(Δ, :u) ? Δ.u : Δ

    if isinplace(cache)
        if cache.f isa BatchIntegralFunction
            error("MooncakeVJP does not yet support BatchIntegralFunction with in-place functions")
        else
            dx = similar(cache.f.integrand_prototype)
            _f = x -> (cache.f(dx, x, p); dx)
            dfdp_ = function (x, p)
                # dx is modified inplace by dfdp/integralfunc_closure_p calls AND the Reverse pass tangent comes externally (from Δ).
                # Therefore, Δ_val is Tangent passed to the pullback AND integralfunc_closure_p must always return dx as Output.
                # i.e. (tangent(output) == Δ_val). Otherwise integralfunc_closure_p only outputs "nothing" and tangent(output) != Δ_val
                integralfunc_closure_p = p -> (cache.f(dx, x, p); dx)
                cache_z = Mooncake.prepare_pullback_cache(integralfunc_closure_p, p)
                z, grads = Mooncake.value_and_pullback!!(cache_z, Δ_val, integralfunc_closure_p, p)
                return grads[2]
            end
            dfdp = IntegralFunction{false}(dfdp_, nothing)
        end
    else
        _f = x -> cache.f(x, p)
        if cache.f isa BatchIntegralFunction
            error("MooncakeVJP does not yet support BatchIntegralFunction")
        else
            dfdp_ = function (x, p)
                integralfunc_closure_p = p -> cache.f(x, p)
                cache_z = Mooncake.prepare_pullback_cache(integralfunc_closure_p, p)
                # Δ_val is integrand function's output sensitivity which we pass into Mooncake's pullback
                z, grads = Mooncake.value_and_pullback!!(cache_z, Δ_val, integralfunc_closure_p, p)
                return grads[2]
            end
            dfdp = IntegralFunction{false}(dfdp_, nothing)
        end
    end
    return dfdp, _f
end

# remove from here once https://github.com/chalk-lab/Mooncake.jl/pull/997 is merged
function Mooncake.increment_and_get_rdata!(
        f::NoFData, r::Tuple{T, T}, t::ChainRulesCore.Tangent{P, Tuple{T, T}}
    ) where {P, T}
    return map((ri, ti) -> increment_and_get_rdata!(f, ri, ti), r, t.backing)
end

function Mooncake.increment_and_get_rdata!(
        f::Tuple{T, T}, r::NoRData, t::ChainRulesCore.Tangent{P, Tuple{T, T}}
    ) where {P, M <: Base.IEEEFloat, T <: AbstractArray{M}}
    increment!!(f, t.backing)
    return r
end

end
