module IntegralsDifferentiationInterfaceExt

using Integrals
using LinearAlgebra: dot
using DifferentiationInterface
using ADTypes: ADTypes, AbstractADType
import ChainRulesCore
import ChainRulesCore: Tangent, NoTangent, ProjectTo
using SciMLLogging: @SciMLMessage

batch_unwrap(x::AbstractArray) = dropdims(x; dims = ndims(x))

# Define rrule for __solvebp when sensealg is an ADTypes backend
function ChainRulesCore.rrule(
        ::typeof(Integrals.__solvebp), cache, alg, sensealg::AbstractADType, domain, p;
        kwargs...
    )
    # Compute the primal value
    out = Integrals.__solvebp_call(cache, alg, sensealg, domain, p; kwargs...)

    # The adjoint computes the integral of the input sensitivities
    function quadrature_adjoint(Δ)
        # Extract the tangent of the integral value from the solution tangent
        # Δ is a NamedTuple representing tangent of IntegralSolution
        Δu = hasproperty(Δ, :u) ? Δ.u : Δ

        # Handle the sensitivity computation using DifferentiationInterface
        if Integrals.isinplace(cache)
            # For in-place integrands, build an out-of-place wrapper for the pullback
            if cache.f isa SciMLBase.BatchIntegralFunction
                dx = similar(
                    cache.f.integrand_prototype,
                    size(cache.f.integrand_prototype)[begin:(end - 1)]..., 1
                )
                _f = x -> (cache.f(dx, x, p); dx)
                dfdp_ = function (x, p)
                    x_ = x isa AbstractArray ? reshape(x, size(x)..., 1) : [x]
                    Δ_ = Δu isa AbstractArray ? reshape(Δu, size(Δu)..., 1) : Δu
                    # Use DI.pullback: pullback(f, backend, x, ty) -> tx
                    return DifferentiationInterface.pullback(
                        p -> (cache.f(dx, x_, p); copy(dx)),
                        sensealg, p, (Δ_,)
                    )[1]
                end
                dfdp = SciMLBase.IntegralFunction{false}(dfdp_, nothing)
            else
                dx = similar(cache.f.integrand_prototype)
                _f = x -> (cache.f(dx, x, p); dx)
                dfdp_ = function (x, p)
                    # Use DI.pullback: pullback(f, backend, x, ty) -> tx
                    return DifferentiationInterface.pullback(
                        p -> (cache.f(dx, x, p); copy(dx)),
                        sensealg, p, (Δu,)
                    )[1]
                end
                dfdp = SciMLBase.IntegralFunction{false}(dfdp_, nothing)
            end
        else
            # Out-of-place integrand
            _f = x -> cache.f(x, p)
            if cache.f isa SciMLBase.BatchIntegralFunction
                dfdp_ = function (x, p)
                    x_ = x isa AbstractArray ? reshape(x, size(x)..., 1) : [x]
                    Δ_ = Δu isa AbstractArray ? reshape(Δu, size(Δu)..., 1) : [Δu]
                    # Use DI.pullback: pullback(f, backend, x, ty) -> tx
                    return DifferentiationInterface.pullback(
                        p -> cache.f(x_, p),
                        sensealg, p, (Δ_,)
                    )[1]
                end
                dfdp = SciMLBase.IntegralFunction{false}(dfdp_, nothing)
            else
                dfdp_ = function (x, p)
                    Δ_ = Δu isa Number ? Δu : only(Δu)
                    # Use DI.pullback: pullback(f, backend, x, ty) -> tx
                    return DifferentiationInterface.pullback(
                        p -> cache.f(x, p),
                        sensealg, p, (Δ_,)
                    )[1]
                end
                dfdp = SciMLBase.IntegralFunction{false}(dfdp_, nothing)
            end
        end

        # Compute dp (gradient w.r.t. p) only if p is not NullParameters
        if p isa SciMLBase.NullParameters
            dp = NoTangent()
        else
            prob = Integrals.build_problem(cache)
            dp_prob = SciMLBase.IntegralProblem(dfdp, prob.domain, prob.p; prob.kwargs...)
            # The infinity transformation was already applied to f so we don't apply it to dfdp
            dp_cache = SciMLBase.init(
                dp_prob,
                alg;
                sensealg = sensealg,
                cache.kwargs...
            )

            project_p = ProjectTo(p)
            dp = project_p(SciMLBase.solve!(dp_cache).u)
        end

        lb, ub = domain
        if lb isa Number
            # Compute boundary gradients using fundamental theorem of calculus
            dlb = cache.f isa SciMLBase.BatchIntegralFunction ? -batch_unwrap(_f([lb])) :
                -_f(lb)
            dub = cache.f isa SciMLBase.BatchIntegralFunction ? batch_unwrap(_f([ub])) :
                _f(ub)
            return (
                NoTangent(),
                NoTangent(),
                NoTangent(),
                NoTangent(),
                Tangent{typeof(domain)}(dot(dlb, Δu), dot(dub, Δu)),
                dp,
            )
        else
            # For multivariate bounds, boundary derivatives are not yet implemented
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), dp)
        end
    end
    return out, quadrature_adjoint
end

end # module
