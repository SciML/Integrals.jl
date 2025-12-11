module IntegralsMooncakeExt
using Mooncake
using LinearAlgebra: dot
using Integrals
using SciMLBase, QuadGK
using Mooncake: @from_chainrules, @zero_derivative, @is_primitive, increment!!, increment_and_get_rdata!, MinimalCtx, rrule!!, NoFData, CoDual, primal, NoRData, zero_fcodual
using Integrals: AbstractIntegralMetaAlgorithm, IntegralProblem
import ChainRulesCore
import ChainRulesCore: Tangent, NoTangent, ProjectTo
using Zygote # use chainrules defined in ZygoteExt

batch_unwrap(x::AbstractArray) = dropdims(x; dims=ndims(x))

@zero_derivative MinimalCtx Tuple{typeof(QuadGK.quadgk),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(QuadGK.cachedrule),Any,Integer}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.checkkwargs),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.isinplace),Vararg}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.init_cacheval),Union{<:SciMLBase.AbstractIntegralAlgorithm,<:AbstractIntegralMetaAlgorithm},Union{<:IntegralProblem,<:SampledIntegralProblem}}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.substitute_f),Union{<:BatchIntegralFunction,<:IntegralFunction},Any,Any,Any}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.substitute_v),Any,Any,Union{<:AbstractVector,<:Number},Union{<:AbstractVector,<:Number}}
@zero_derivative MinimalCtx Tuple{typeof(Integrals.substitute_bv),Any,AbstractArray,Union{<:AbstractVector,<:Number},Union{<:AbstractVector,<:Number}}

# @from_chainrules MinimalCtx Tuple{Type{IntegralProblem{iip}},Any,Any,Any} where {iip} true
@is_primitive MinimalCtx Tuple{Type{IntegralProblem{iip}},Any,Any,Any} where {iip}
function Mooncake.rrule!!(::CoDual{Type{IntegralProblem{iip}}}, f::CoDual, domain::CoDual, p::CoDual; kwargs...) where {iip}
    f_prim, domain_prim, p_prim = map(primal, (f, domain, p))
    prob = IntegralProblem{iip}(f_prim, domain_prim, p_prim; kwargs...)

    function IntegralProblem_iip_pullback(Δ)
        data = Δ isa NoRData ? Δ : Δ.data
        ddomain = hasproperty(data, :domain) ? data.domain : NoRData()
        dp = hasproperty(data, :p) ? data.p : NoRData()
        dkwargs = Δ isa NoRData ? Δ : data.kwargs

        # domain is always a Tuple, so it always has NoFData
        # below conditional is in case p is an Array or similar
        if Mooncake.rdata_type(typeof(p_prim)) == NoRData()
            p.dx .+= dp
            grad_p = NoRData()
        else
            grad_p = dp
        end

        return NoRData(), NoRData(), ddomain, grad_p, dkwargs
    end
    return zero_fcodual(prob), IntegralProblem_iip_pullback
end

@is_primitive MinimalCtx Tuple{typeof(SciMLBase.build_solution),IntegralProblem,Any,Any,Any}
function Mooncake.rrule!!(::CoDual{typeof(SciMLBase.build_solution)}, prob::CoDual{IntegralProblem}, alg::CoDual, u::CoDual, resid::CoDual; kwargs)
    kwargs_fp = map(primal, kwargs)
    function pb!!(Δ)
        return NoRData(),
        NoRData(),
        NoRData(),
        Δ,
        NoRData()
    end
    return SciMLBase.build_solution(prob.primal, alg.primal, u.primal, resid.primal; kwargs_fp...), pb!!
end

# Mooncake does not need chainrule for evaluate! as it supports mutation.
@from_chainrules MinimalCtx Tuple{Type{IntegralProblem},Any,Any,Any} true
@from_chainrules MinimalCtx Tuple{typeof(SciMLBase.build_solution),IntegralProblem,Any,Any,Any} true
@from_chainrules MinimalCtx Tuple{typeof(Integrals.u2t),Any,Any} true
@from_chainrules MinimalCtx Tuple{typeof(Integrals.__solvebp),Any,Any,Any,Any,Any} true

function ChainRulesCore.rrule(::typeof(Integrals.__solvebp), cache, alg, sensealg, domain,
    p;
    kwargs...)
    # TODO: integrate the primal and dual in the same call to the quadrature library
    out = Integrals.__solvebp_call(cache, alg, sensealg, domain, p; kwargs...)

    # the adjoint will be the integral of the input sensitivities, so it maps the
    # sensitivity of the output to an object of the type of the parameters
    function quadrature_adjoint(Δ)
        # https://juliadiff.org/ChainRulesCore.jl/dev/design/many_tangents.html#manytypes
        if sensealg.vjp isa Integrals.ZygoteVJP
            if isinplace(cache)
                # zygote doesn't support mutation, so we build an oop pullback
                if cache.f isa BatchIntegralFunction
                    dx = similar(cache.f.integrand_prototype,
                        size(cache.f.integrand_prototype)[begin:(end-1)]..., 1)
                    _f = x -> (cache.f(dx, x, p); dx)
                    # TODO: let the user pass a batched jacobian so we can return a BatchIntegralFunction
                    dfdp_ = function (x, p)
                        x_ = x isa AbstractArray ? reshape(x, size(x)..., 1) : [x]
                        z, back = Zygote.pullback(p) do p
                            _dx = Zygote.Buffer(dx)
                            cache.f(_dx, x_, p)
                            copy(_dx)
                        end
                        return back(z .= (Δ isa AbstractArray ? reshape(Δ, size(Δ)..., 1) :
                                          Δ))[1]
                    end
                    dfdp = IntegralFunction{false}(dfdp_, nothing)
                else
                    dx = similar(cache.f.integrand_prototype)
                    _f = x -> (cache.f(dx, x, p); dx)
                    dfdp_ = function (x, p)
                        _, back = Zygote.pullback(p) do p
                            _dx = Zygote.Buffer(dx)
                            cache.f(_dx, x, p)
                            copy(_dx)
                        end
                        back(Δ)[1]
                    end
                    dfdp = IntegralFunction{false}(dfdp_, nothing)
                end
            else
                _f = x -> cache.f(x, p)
                if cache.f isa BatchIntegralFunction
                    # TODO: let the user pass a batched jacobian so we can return a BatchIntegralFunction
                    dfdp_ = function (x, p)
                        x_ = x isa AbstractArray ? reshape(x, size(x)..., 1) : [x]
                        z, back = Zygote.pullback(p -> cache.f(x_, p), p)
                        return back(Δ isa AbstractArray ? reshape(Δ, size(Δ)..., 1) : [Δ])[1]
                    end
                    dfdp = IntegralFunction{false}(dfdp_, nothing)
                else
                    dfdp_ = function (x, p)
                        z, back = Zygote.pullback(p -> cache.f(x, p), p)
                        back(z isa Number ? only(Δ) : Δ)[1]
                    end
                    dfdp = IntegralFunction{false}(dfdp_, nothing)
                end
            end
        elseif sensealg.vjp isa Integrals.MooncakeVJP
            _f = x -> cache.f(x, p)
            if cache.f isa BatchIntegralFunction
                # TODO: let the user pass a batched jacobian so we can return a BatchIntegralFunction
                dfdp_ = function (x, p)
                    x_ = x isa AbstractArray ? reshape(x, size(x)..., 1) : [x]
                    clos = p -> cache.f(x_, p)
                    cache_z = Mooncake.prepare_pullback_cache(clos, p)
                    z, grads = Mooncake.value_and_pullback!!(cache_z, [Δ.u], clos, p)
                    return grads[2]
                end
                dfdp = IntegralFunction{false}(dfdp_, nothing)
            else
                dfdp_ = function (x, p)
                    clos = p -> cache.f(x, p)
                    cache_z = Mooncake.prepare_pullback_cache(clos, p)
                    # Δ.u is integrand function's output sensitivity which we pass into Mooncake's pullback
                    z, grads = Mooncake.value_and_pullback!!(cache_z, Δ.u, clos, p)
                    return grads[2]
                end
                dfdp = IntegralFunction{false}(dfdp_, nothing)
            end
        elseif sensealg.vjp isa Integrals.ReverseDiffVJP
            error("TODO")
        end

        prob = Integrals.build_problem(cache)
        # dp_prob = remake(prob, f = dfdp)  # fails because we change iip
        dp_prob = IntegralProblem(dfdp, prob.domain, prob.p; prob.kwargs...)
        # the infinity transformation was already applied to f so we don't apply it to dfdp
        dp_cache = init(dp_prob,
            alg;
            sensealg=sensealg,
            cache.kwargs...)

        project_p = ProjectTo(p)
        dp = project_p(solve!(dp_cache).u)
    
        # Because Mooncake tangent structure vs Zygote, Chainrules, ReverseDiff
        du_adj = sensealg.vjp isa Integrals.MooncakeVJP ? Δ.u : Δ

        lb, ub = domain
        if lb isa Number
            # TODO replace evaluation at endpoint (which anyone can do without Integrals.jl)
            # with integration of dfdx uing the same quadrature
            dlb = cache.f isa BatchIntegralFunction ? -batch_unwrap(_f([lb])) : -_f(lb)
            dub = cache.f isa BatchIntegralFunction ? batch_unwrap(_f([ub])) : _f(ub)
            return (NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                Tangent{typeof(domain)}(dot(dlb, du_adj), dot(dub,  du_adj)),
                dp)
        else
            # we need to compute 2*length(lb) integrals on the faces of the hypercube, as we
            # can see from writing the multidimensional integral as an iterated integral
            # alternatively we can use Stokes' theorem to replace the integral on the
            # boundary with a volume integral of the flux of the integrand
            # ∫∂Ω ω = ∫Ω dω, which would be better since we won't have to change the
            # dimensionality of the integral or the quadrature used (such as quadratures
            # that don't evaluate points on the boundaries) and it could be generalized to
            # other kinds of domains. The only question is to determine ω in terms of f and
            # the deformation of the surface (e.g. consider integral over an ellipse and
            # asking for the derivative of the result w.r.t. the semiaxes of the ellipse)
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), dp)
        end
    end
    out, quadrature_adjoint
end

function Mooncake.increment_and_get_rdata!(
    f::Tuple{Vector{Float64},Vector{Float64}},
    r::Mooncake.NoRData,
    t::Tangent{Any,Tuple{Vector{Float64},Vector{Float64}}},
)
    Mooncake.increment!!(f[1], t[1])
    Mooncake.increment!!(f[2], t[2])
    return NoRData()
end

function Mooncake.increment_and_get_rdata!(
    f::NoFData, r::Tuple{T,T}, t::Union{Tangent{Any,Tuple{T,T}},Tangent{Tuple{T,T},Tuple{T,T}}}
) where {T<:Base.IEEEFloat}
    return r .+ t.backing
end

function Mooncake.increment_and_get_rdata!(
    f::NoFData,
    r::T,
    t::Tangent{Any,@NamedTuple{u::T, resid::T, prob::Tangent{Any,@NamedTuple{f::NoTangent, domain::Tangent{Any,Tuple{T,T}}, p::T, kwargs::NoTangent}},
        alg::NoTangent, retcode::NoTangent, chi::NoTangent, stats::NoTangent}}) where T<:Number
    return Mooncake.increment!!(t.u, r)
end

end