module IntegralsZygoteExt
using Integrals
if isdefined(Base, :get_extension)
    using Zygote
    import ChainRulesCore
    import ChainRulesCore: NoTangent
else
    using ..Zygote
    import ..Zygote.ChainRulesCore
    import ..Zygote.ChainRulesCore: NoTangent
end
ChainRulesCore.@non_differentiable Integrals.checkkwargs(kwargs...)

function ChainRulesCore.rrule(::typeof(Integrals.__solvebp), prob, alg, sensealg, lb, ub, p;
                              kwargs...)
    out = Integrals.__solvebp_call(prob, alg, sensealg, lb, ub, p; kwargs...)
    function quadrature_adjoint(Δ)
        y = typeof(Δ) <: Array{<:Number, 0} ? Δ[1] : Δ
        if isinplace(prob)
            dx = zeros(prob.nout)
            _f = x -> prob.f(dx, x, p)
            if sensealg.vjp isa Integrals.ZygoteVJP
                dfdp = function (dx, x, p)
                    _, back = Zygote.pullback(p) do p
                        _dx = Zygote.Buffer(x, prob.nout, size(x, 2))
                        prob.f(_dx, x, p)
                        copy(_dx)
                    end

                    z = zeros(size(x, 2))
                    for idx in 1:size(x, 2)
                        z[1] = 1
                        dx[:, idx] = back(z)[1]
                        z[idx] = 0
                    end
                end
            elseif sensealg.vjp isa Integrals.ReverseDiffVJP
                error("TODO")
            end
        else
            _f = x -> prob.f(x, p)
            if sensealg.vjp isa Integrals.ZygoteVJP
                if prob.batch > 0
                    dfdp = function (x, p)
                        _, back = Zygote.pullback(p -> prob.f(x, p), p)

                        out = zeros(length(p), size(x, 2))
                        z = zeros(size(x, 2))
                        for idx in 1:size(x, 2)
                            z[idx] = 1
                            out[:, idx] = back(z)[1]
                            z[idx] = 0
                        end
                        out
                    end
                else
                    dfdp = function (x, p)
                        _, back = Zygote.pullback(p -> prob.f(x, p), p)
                        back(y)[1]
                    end
                end

            elseif sensealg.vjp isa Integrals.ReverseDiffVJP
                error("TODO")
            end
        end

        dp_prob = remake(prob, f = dfdp, lb = lb, ub = ub, p = p, nout = length(p))

        if p isa Number
            dp = Integrals.__solvebp_call(dp_prob, alg, sensealg, lb, ub, p; kwargs...)[1]
        else
            dp = Integrals.__solvebp_call(dp_prob, alg, sensealg, lb, ub, p; kwargs...).u
        end

        if lb isa Number
            dlb = -_f(lb)
            dub = _f(ub)
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), dlb, dub, dp)
        else
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                    NoTangent(), dp)
        end
    end
    out, quadrature_adjoint
end

Zygote.@adjoint function Zygote.literal_getproperty(sol::SciMLBase.IntegralSolution,
                                                    ::Val{:u})
    sol.u, Δ -> (SciMLBase.build_solution(sol.prob, sol.alg, Δ, sol.resid),)
end
end
