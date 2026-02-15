module IntegralsMooncakeExt
using Mooncake
using LinearAlgebra: dot
using Integrals, SciMLBase, QuadGK
using Mooncake: @from_chainrules, @is_primitive, increment!!, MinimalCtx, rrule!!, NoFData,
    CoDual, primal, NoRData, zero_fcodual
import Mooncake: increment_and_get_rdata!, @zero_derivative
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

# @from_chainrules MinimalCtx Tuple{Type{IntegralProblem{iip}},Any,Any,Any} where {iip} true
@is_primitive MinimalCtx Tuple{Type{IntegralProblem{iip}}, Any, Any, Any} where {iip}
function Mooncake.rrule!!(
        ::CoDual{Type{IntegralProblem{iip}}}, f::CoDual,
        domain::CoDual, p::CoDual; kwargs...
    ) where {iip}
    f_prim, domain_prim, p_prim = map(primal, (f, domain, p))
    prob = IntegralProblem{iip}(f_prim, domain_prim, p_prim; kwargs...)

    function IntegralProblem_iip_pullback(Δ)
        data = Δ isa NoRData ? Δ : Δ.data
        ddomain = hasproperty(data, :domain) ? data.domain : NoRData()
        dp = hasproperty(data, :p) ? data.p : NoRData()
        dkwargs = hasproperty(Δ, :kwargs) ? data.kwargs : NoRData()

        # domain is always a Tuple, so it always has NoFData
        # below conditional is in case p is an Array or similar
        if Mooncake.rdata_type(typeof(p_prim)) == NoRData()
            Mooncake.increment!!(p.dx, dp)
            grad_p = NoRData()
        else
            grad_p = dp
        end

        return NoRData(), NoRData(), ddomain, grad_p, dkwargs
    end
    return zero_fcodual(prob), IntegralProblem_iip_pullback
end

# Mooncake does not need chainrule for evaluate! as it supports mutation.
@from_chainrules MinimalCtx Tuple{Type{IntegralProblem}, Any, Any, Any} true
@from_chainrules MinimalCtx Tuple{typeof(Integrals.u2t), Any, Any} true
@from_chainrules MinimalCtx Tuple{
    typeof(SciMLBase.build_solution), IntegralProblem, Any, Any, Any,
} true

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

# Internal Mooncake overloads to accommodate IntegralSolution etc. Struct's Tangent Types.
# Allows clear translation from ChainRules -> Mooncake's tangent.
function Mooncake.increment_and_get_rdata!(
        f::NoFData, r::Tuple{T, T},
        t::Union{Tangent{Tuple{T, T}, Tuple{T, T}}, Tangent{Any, Tuple{Float64, Float64}}}
    ) where {T <: Base.IEEEFloat}
    return r .+ t.backing
end

function Mooncake.increment_and_get_rdata!(
        f::Tuple{Vector{T}, Vector{T}},
        r::NoRData,
        t::Tangent{Any, Tuple{Vector{T}, Vector{T}}}
    ) where {T <: Base.IEEEFloat}
    Mooncake.increment!!(f, t.backing)
    return NoRData()
end

# sol.u & p are single scalar values, domain (lb,ub) is single/multi - variate.
function Mooncake.increment_and_get_rdata!(
        f::NoFData,
        r::T,
        t::Tangent{
            Any,
            @NamedTuple{
                u::T,
                resid::R,
                prob::Tangent{
                    Any,
                    @NamedTuple{
                        f::NoTangent,
                        domain::Tangent{Any, Tuple{M, M}},
                        p::P,
                        kwargs::NoTangent,
                    }
                },
                alg::A,
                retcode::NoTangent,
                chi::NoTangent,
                stats::NoTangent,
            }
        }
    ) where {
        T <: Base.IEEEFloat,
        R <: Union{NoTangent, T},
        P <: Union{T, Vector{T}},
        M <: Union{T, Vector{T}},
        A <: Union{
            NoTangent,
            Tangent{
                Any,
                @NamedTuple{
                    nodes::Vector{T},
                    weights::Vector{T},
                    subintervals::NoTangent,
                }
            }
        }
    }
    # rdata component of t + r (u field)
    return Mooncake.increment_and_get_rdata!(f, r, t.u)
end

# sol.u is vector valued, p is scalar/vector valued, domain can be single/multi - variate
#  resid can be single/vector valued. For inplace integrals (iip true) : included integrand_prototype field in typeof{prob.f}
function Mooncake.increment_and_get_rdata!(
        f::Vector{T},
        r::NoRData,
        t::Union{
            Tangent{
                Any,
                @NamedTuple{
                    u::Vector{T},
                    resid::R,
                    prob::Tangent{
                        Any,
                        @NamedTuple{
                            f::F,
                            domain::Tangent{Any, M},
                            p::P,
                            kwargs::NoTangent,
                        }
                    },
                    alg::A,
                    retcode::NoTangent,
                    chi::NoTangent,
                    stats::NoTangent,
                }
            }
        }
    ) where {
        T <: Base.IEEEFloat,
        R <: Union{NoTangent, T, Vector{T}},
        P <: Union{T, Vector{T}},
        M <: Union{Tuple{T, T}, Tuple{Vector{T}, Vector{T}}},
        F <: Union{
            NoTangent,
            Tangent{
                Any,
                @NamedTuple{
                    f::NoTangent,
                    integrand_prototype::Vector{T},
                }
            }
        },
        A <: Union{
            NoTangent,
            Tangent{
                Any,
                @NamedTuple{
                    nodes::Vector{T},
                    weights::Vector{T},
                    subintervals::NoTangent,
                }
            }
        }
    }
    Mooncake.increment!!(f, t.u)
    # rdata component(t) + r
    return t.prob.domain
end

function Mooncake.increment!!(
        ::Mooncake.NoRData,
        y::Tangent{Any, Y}
    ) where {
        T <: Base.IEEEFloat, Y <: Union{Tuple{T, T}, Tuple{Vector{T}, Vector{T}}},
    }
    return NoRData()
end

function Mooncake.increment!!(
        x::Tangent{Any, Y},
        ::Mooncake.NoRData
    ) where {
        T <: Base.IEEEFloat, Y <: Union{Tuple{T, T}, Tuple{Vector{T}, Vector{T}}},
    }
    return x
end

end
