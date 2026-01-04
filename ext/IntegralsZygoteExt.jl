module IntegralsZygoteExt
using LinearAlgebra: dot
using Integrals
using Zygote
import ChainRulesCore
import ChainRulesCore: Tangent, NoTangent, ProjectTo
using Mooncake # call __solve_bp's chainrule from Mooncake's extension

ChainRulesCore.@non_differentiable Integrals.checkkwargs(kwargs...)
ChainRulesCore.@non_differentiable Integrals.isinplace(f, args...)    # fixes #99
ChainRulesCore.@non_differentiable Integrals.init_cacheval(alg, prob)
ChainRulesCore.@non_differentiable Integrals.substitute_f(args...) # use ∂f/∂p instead
ChainRulesCore.@non_differentiable Integrals.substitute_v(args...) # TODO for ∂f/∂u
ChainRulesCore.@non_differentiable Integrals.substitute_bv(args...) # TODO for ∂f/∂u

function ChainRulesCore.rrule(::Type{<:IntegralProblem}, f, domain, p; kwargs...)
    prob = IntegralProblem(f, domain, p; kwargs...)
    function IntegralProblem_pullback(Δ)
        ddomain = hasproperty(Δ, :domain) ? Δ.domain : NoTangent()
        dp = hasproperty(Δ, :p) ? Δ.p : NoTangent()
        return NoTangent(), NoTangent(), ddomain, dp
    end
    return prob, IntegralProblem_pullback
end

function ChainRulesCore.rrule(
        ::Type{IntegralProblem{iip}}, f, domain, p; kwargs...
    ) where {iip}
    prob = IntegralProblem{iip}(f, domain, p; kwargs...)
    function IntegralProblem_iip_pullback(Δ)
        ddomain = hasproperty(Δ, :domain) ? Δ.domain : NoTangent()
        dp = hasproperty(Δ, :p) ? Δ.p : NoTangent()
        return NoTangent(), NoTangent(), ddomain, dp
    end
    return prob, IntegralProblem_iip_pullback
end

# TODO move this adjoint to SciMLBase
function ChainRulesCore.rrule(
        ::typeof(SciMLBase.build_solution), prob::IntegralProblem, alg, u, resid; kwargs...
    )
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
    return out, Δ -> (NoTangent(), NoTangent(), back(Δ)...)
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

Zygote.@adjoint function Zygote.literal_getproperty(
        sol::SciMLBase.IntegralSolution,
        ::Val{:u}
    )
    sol.u, Δ -> (SciMLBase.build_solution(sol.prob, sol.alg, Δ, sol.resid),)
end
end
