module IntegralsZygoteExt
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
            dx = similar(cache.f.integrand_prototype)
            _f = x -> cache.f(dx, x, p)
            if sensealg.vjp isa Integrals.ZygoteVJP
                if cache.f isa BatchIntegralFunction
                    # TODO: let the user pass a batched jacobian so we can return a BatchIntegralFunction
                    dfdp_ = function (x, p)
                        x_ = x isa AbstractArray ? reshape(x, size(x)..., 1) : [x]
                        z, back = Zygote.pullback(p) do p
                            _dx = Zygote.Buffer(dx, size(dx)[begin:(end - 1)]..., 1)
                            cache.f(_dx, x_, p)
                            copy(_dx)
                        end
                        return back(z .= (Δ isa AbstractArray ? reshape(Δ, size(Δ)..., 1) :
                                          [Δ]))[1]
                    end
                    dfdp = IntegralFunction{false}(dfdp_, nothing)
                else
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
                        return back(z .= (Δ isa AbstractArray ? reshape(Δ, size(Δ)..., 1) :
                                          [Δ]))[1]
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
        #=
        if isinplace(cache)
            dx = zeros(cache.nout)
            _f = x -> cache.f(dx, x, p)
            if sensealg.vjp isa Integrals.ZygoteVJP
                dfdp = function (dx, x, p)
                    z, back = Zygote.pullback(p) do p
                        _dx = cache.nout == 1 ?
                              Zygote.Buffer(dx, eltype(y), size(x, ndims(x))) :
                              Zygote.Buffer(dx, eltype(y), cache.nout, size(x, ndims(x)))
                        cache.f(_dx, x, p)
                        copy(_dx)
                    end
                    z .= zero(eltype(z))
                    for idx in 1:size(x, ndims(x))
                        z isa Vector ? (z[idx] = y) : (z[:, idx] .= y)
                        dx[:, idx] .= back(z)[1]
                        z isa Vector ? (z[idx] = zero(eltype(z))) :
                        (z[:, idx] .= zero(eltype(z)))
                    end
                end
            elseif sensealg.vjp isa Integrals.ReverseDiffVJP
                error("TODO")
            end
        else
            _f = x -> cache.f(x, p)
            if sensealg.vjp isa Integrals.ZygoteVJP
                if cache.batch > 0
                    dfdp = function (x, p)
                        z, back = Zygote.pullback(p -> cache.f(x, p), p)
                        # messy, there are 4 cases, some better in forward mode than reverse
                        # 1: length(y) == 1 and length(p) == 1
                        # 2: length(y) >  1 and length(p) == 1
                        # 3: length(y) == 1 and length(p) >  1
                        # 4: length(y) >  1 and length(p) >  1

                        z .= zero(eltype(z))
                        out = zeros(eltype(p), size(p)..., size(x, ndims(x)))
                        for idx in 1:size(x, ndims(x))
                            z isa Vector ? (z[idx] = y) : (z[:, idx] .= y)
                            out isa Vector ? (out[idx] = back(z)[1]) :
                            (out[:, idx] .= back(z)[1])
                            z isa Vector ? (z[idx] = zero(y)) :
                            (z[:, idx] .= zero(eltype(y)))
                        end
                        out
                    end
                else
                    dfdp = function (x, p)
                        _, back = Zygote.pullback(p -> cache.f(x, p), p)
                        back(y)[1]
                    end
                end

            elseif sensealg.vjp isa Integrals.ReverseDiffVJP
                error("TODO")
            end
        end
        =#
        prob = Integrals.build_problem(cache)
        # dp_prob = remake(prob, f = dfdp)  # fails because we change iip
        dp_prob = IntegralProblem(dfdp, prob.domain, prob.p; prob.kwargs...)
        # the infinity transformation was already applied to f so we don't apply it to dfdp
        dp_cache = init(dp_prob,
            alg;
            sensealg = sensealg,
            do_inf_transformation = Val(false),
            cache.kwargs...)

        project_p = ProjectTo(p)
        dp = project_p(Integrals.__solvebp_call(dp_cache,
            alg,
            sensealg,
            domain,
            p;
            kwargs...).u)

        lb, ub = domain
        if lb isa Number
            dlb = cache.f isa BatchIntegralFunction ? -_f([lb]) : -_f(lb)
            dub = cache.f isa BatchIntegralFunction ? _f([ub]) : _f(ub)
            return (NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                Tangent{typeof(domain)}(dlb, dub),
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

Zygote.@adjoint function Zygote.literal_getproperty(sol::SciMLBase.IntegralSolution,
        ::Val{:u})
    sol.u, Δ -> (SciMLBase.build_solution(sol.prob, sol.alg, Δ, sol.resid),)
end
end
