module IntegralsForwardDiffExt
using Integrals
using Integrals: set_f, set_p, build_problem
isdefined(Base, :get_extension) ? (using ForwardDiff) : (using ..ForwardDiff)
### Forward-Mode AD Intercepts

# Direct AD on solvers with QuadGK and HCubature
function Integrals.__solvebp(cache, alg::QuadGKJL, sensealg, lb, ub,
                             p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                             kwargs...) where {T, V, P, N}
    Integrals.__solvebp_call(cache, alg, sensealg, lb, ub, p; kwargs...)
end

function Integrals.__solvebp(cache, alg::HCubatureJL, sensealg, lb, ub,
                             p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                             kwargs...) where {T, V, P, N}
    Integrals.__solvebp_call(cache, alg, sensealg, lb, ub, p; kwargs...)
end

# Manually split for the pushforward
function Integrals.__solvebp(cache, alg, sensealg, lb, ub,
                             p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                             kwargs...) where {T, V, P, N}
    primal = Integrals.__solvebp_call(cache, alg, sensealg, lb, ub, ForwardDiff.value.(p);
                                      kwargs...)

    nout = cache.nout * P

    if isinplace(cache)
        dfdp = function (out, x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            if cache.batch > 0
                dx = similar(dualp, cache.nout, size(x, 2))
            else
                dx = similar(dualp, cache.nout)
            end
            cache.f(dx, x, dualp)

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
            ys = cache.f(x, dualp)
            if cache.batch > 0
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

    dp_cache = set_p(set_f(cache, dfdp, nout), rawp)
    dual = Integrals.__solvebp_call(dp_cache, alg, sensealg, lb, ub, rawp; kwargs...)

    res = similar(p, cache.nout)
    partials = reinterpret(typeof(first(res).partials), dual.u)
    for idx in eachindex(res)
        res[idx] = ForwardDiff.Dual{T, V, P}(primal.u[idx], partials[idx])
    end
    if primal.u isa Number
        res = first(res)
    end
    SciMLBase.build_solution(build_problem(cache), alg, res, primal.resid)
end
end
