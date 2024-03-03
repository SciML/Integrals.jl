module IntegralsZygoteExt
using LinearAlgebra: dot
using Integrals
if isdefined(Base, :get_extension)
    using Zygote
    import ChainRulesCore
    import ChainRulesCore: Tangent, NoTangent, ProjectTo
else
    using ..Zygote
    import ..Zygote.ChainRulesCore
    import ..Zygote.ChainRulesCore: Tangent, NoTangent, ProjectTo
end
ChainRulesCore.@non_differentiable Integrals.checkkwargs(kwargs...)
ChainRulesCore.@non_differentiable Integrals.isinplace(f, args...)    # fixes #99
ChainRulesCore.@non_differentiable Integrals.init_cacheval(alg, prob)
ChainRulesCore.@non_differentiable Integrals.substitute_f(args...) # use ∂f/∂p instead
ChainRulesCore.@non_differentiable Integrals.substitute_v(args...) # TODO for ∂f/∂u
ChainRulesCore.@non_differentiable Integrals.substitute_bv(args...) # TODO for ∂f/∂u

# TODO move this adjoint to SciMLBase
function ChainRulesCore.rrule(
        ::typeof(SciMLBase.build_solution), prob::IntegralProblem, alg, u, resid; kwargs...)
    function build_integral_solution_pullback(Δ)
        return NoTangent(), NoTangent(), NoTangent(), Δ, NoTangent()
    end
    return SciMLBase.build_solution(prob, alg, u, resid; kwargs...),
    build_integral_solution_pullback
end

function ChainRulesCore.rrule(::typeof(Integrals._evaluate!), f, y, u, p)
    out, back = Zygote.pullback(y, u, p) do y, u, p
        b = Zygote.Buffer(y)
        f(b, u, p)
        return copy(b)
    end
    out, Δ -> (NoTangent(), NoTangent(), back(Δ)...)
end

function ChainRulesCore.rrule(::typeof(Integrals.u2t), lb, ub)
    tlb, tub = out = Integrals.u2t(lb, ub)
    function u2t_pullback(Δ)
        _, lbjac = Integrals.t2ujac(tlb, lb, ub)
        _, ubjac = Integrals.t2ujac(tub, lb, ub)
        return NoTangent(), Δ[1] / lbjac, Δ[2] / ubjac
    end
    return out, u2t_pullback
end

function ChainRulesCore.rrule(::typeof(Integrals.__solvebp), cache, alg, sensealg, domain,
        p;
        kwargs...)
    # TODO: integrate the primal and dual in the same call to the quadrature library
    out = Integrals.__solvebp_call(cache, alg, sensealg, domain, p; kwargs...)

    # the adjoint will be the integral of the input sensitivities, so it maps the
    # sensitivity of the output to an object of the type of the parameters
    function quadrature_adjoint(Δ)
        # https://juliadiff.org/ChainRulesCore.jl/dev/design/many_tangents.html#manytypes
        if isinplace(cache)
            # zygote doesn't support mutation, so we build an oop pullback
            if sensealg.vjp isa Integrals.ZygoteVJP
                if cache.f isa BatchIntegralFunction
                    dx = similar(cache.f.integrand_prototype,
                        size(cache.f.integrand_prototype)[begin:(end - 1)]..., 1)
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
            elseif sensealg.vjp isa Integrals.ReverseDiffVJP
                error("TODO")
            end
        else
            _f = x -> cache.f(x, p)
            if sensealg.vjp isa Integrals.ZygoteVJP
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
            elseif sensealg.vjp isa Integrals.ReverseDiffVJP
                error("TODO")
            end
        end

        prob = Integrals.build_problem(cache)
        # dp_prob = remake(prob, f = dfdp)  # fails because we change iip
        dp_prob = IntegralProblem(dfdp, prob.domain, prob.p; prob.kwargs...)
        # the infinity transformation was already applied to f so we don't apply it to dfdp
        dp_cache = init(dp_prob,
            alg;
            sensealg = sensealg,
            cache.kwargs...)

        project_p = ProjectTo(p)
        dp = project_p(solve!(dp_cache).u)

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
                Tangent{typeof(domain)}(dot(dlb, Δ), dot(dub, Δ)),
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

batch_unwrap(x::AbstractArray) = dropdims(x; dims = ndims(x))

Zygote.@adjoint function Zygote.literal_getproperty(sol::SciMLBase.IntegralSolution,
        ::Val{:u})
    sol.u, Δ -> (SciMLBase.build_solution(sol.prob, sol.alg, Δ, sol.resid),)
end
end
