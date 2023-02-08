module IntegralsForwardDiffExt
using Integrals
isdefined(Base, :get_extension) ? (using ForwardDiff) : (using ..ForwardDiff)
### Forward-Mode AD Intercepts

# Direct AD on solvers with QuadGK and HCubature
function Integrals.__solvebp(prob, alg::QuadGKJL, sensealg, lb, ub,
                             p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                             kwargs...) where {T, V, P, N}
    Integrals.__solvebp_call(prob, alg, sensealg, lb, ub, p; kwargs...)
end

function Integrals.__solvebp(prob, alg::HCubatureJL, sensealg, lb, ub,
                             p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                             kwargs...) where {T, V, P, N}
    Integrals.__solvebp_call(prob, alg, sensealg, lb, ub, p; kwargs...)
end

# Manually split for the pushforward
function Integrals.__solvebp(prob, alg, sensealg, lb, ub,
                             p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                             kwargs...) where {T, V, P, N}
    primal = Integrals.__solvebp_call(prob, alg, sensealg, lb, ub, ForwardDiff.value.(p);
                                      kwargs...)

    nout = prob.nout * P

    if isinplace(prob)
        dfdp = function (out, x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            if prob.batch > 0
                dx = similar(dualp, prob.nout, size(x, 2))
            else
                dx = similar(dualp, prob.nout)
            end
            prob.f(dx, x, dualp)

            ys = reinterpret(ForwardDiff.Dual{T, V, P}, dx)
            idx = 0
            for y in ys
                for p in ForwardDiff.partials(y)
                    out[idx += 1] = p
                end
            end
            return out
        end
    else
        dfdp = function (x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            ys = prob.f(x, dualp)
            if prob.batch > 0
                out = similar(p, V, nout, size(x, 2))
            else
                out = similar(p, V, nout)
            end

            idx = 0
            for y in ys
                for p in ForwardDiff.partials(y)
                    out[idx += 1] = p
                end
            end

            return out
        end
    end
    rawp = copy(reinterpret(V, p))

    dp_prob = IntegralProblem(dfdp, lb, ub, rawp; nout = nout, batch = prob.batch,
                              kwargs...)
    dual = Integrals.__solvebp_call(dp_prob, alg, sensealg, lb, ub, rawp; kwargs...)
    res = similar(p, prob.nout)
    partials = reinterpret(typeof(first(res).partials), dual.u)
    for idx in eachindex(res)
        res[idx] = ForwardDiff.Dual{T, V, P}(primal.u[idx], partials[idx])
    end
    if primal.u isa Number
        res = first(res)
    end
    SciMLBase.build_solution(prob, alg, res, primal.resid)
end
end
