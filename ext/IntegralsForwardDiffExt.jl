module IntegralsForwardDiffExt
using Integrals
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
                dx = cache.nout == 1 ? similar(dualp, size(x, ndims(x))) : similar(dualp, cache.nout, size(x, ndims(x)))
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
                out = similar(p, V, nout, size(x, ndims(x)))
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

    prob = Integrals.build_problem(cache)
    dp_prob = remake(prob, f = dfdp, nout = nout, p = rawp)
    # the infinity transformation was already applied to f so we don't apply it to dfdp
    dp_cache = init(dp_prob,
        alg;
        sensealg = sensealg,
        do_inf_transformation = Val(false),
        cache.kwargs...)
    dual = Integrals.__solvebp_call(dp_cache, alg, sensealg, lb, ub, rawp; kwargs...)

    res = similar(p, cache.nout)
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
