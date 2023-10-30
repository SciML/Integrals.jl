module IntegralsForwardDiffExt
using Integrals
isdefined(Base, :get_extension) ? (using ForwardDiff) : (using ..ForwardDiff)
### Forward-Mode AD Intercepts

# Direct AD on solvers with QuadGK and HCubature
function Integrals.__solvebp(cache, alg::QuadGKJL, sensealg, domain,
        p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
        kwargs...) where {T, V, P, N}
    Integrals.__solvebp_call(cache, alg, sensealg, domain, p; kwargs...)
end

function Integrals.__solvebp(cache, alg::HCubatureJL, sensealg, domain,
        p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
        kwargs...) where {T, V, P, N}
    Integrals.__solvebp_call(cache, alg, sensealg, domain, p; kwargs...)
end

# Manually split for the pushforward
function Integrals.__solvebp(cache, alg, sensealg, domain,
        p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
        kwargs...) where {T, V, P, N}

    # we need the output type to avoid perturbation confusion while unwrapping nested duals
    # We compute a vector-valued integral of the primal and dual simultaneously
    if isinplace(cache)
        elt = eltype(cache.f.integrand_prototype)
        DT = replace_dualvaltype(eltype(p), elt)
        len = duallen(p)
        dual_prototype = similar(cache.f.integrand_prototype,
            len,
            size(cache.f.integrand_prototype)...)

        dfdp_ = function (out, x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            dout = reinterpret(reshape, DT, out)
            cache.f(dout, x, dualp)
            return out
        end
        dfdp = if cache.f isa BatchIntegralFunction
            BatchIntegralFunction{true}(dfdp_, dual_prototype)
        else
            IntegralFunction{true}(dfdp_, dual_prototype)
        end
    else
        lb, ub = domain
        mid = (lb + ub) / 2
        y = if cache.f isa BatchIntegralFunction
            mid isa Number ? cache.f(eltype(mid)[], p) :
            cache.f(Matrix{eltype(mid)}(undef, length(mid), 0), p)
        else
            cache.f(mid, p)
        end
        DT = y isa AbstractArray ? eltype(y) : typeof(y)
        elt = unwrap_dualvaltype(DT)

        dfdp_ = function (x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            ys = cache.f(x, dualp)
            ys_ = ys isa AbstractArray ? ys : [ys]
            # we need to reshape in order for batching to be consistent
            return reinterpret(reshape, elt, ys_)
        end
        dfdp = if cache.f isa BatchIntegralFunction
            BatchIntegralFunction{false}(dfdp_, nothing)
        else
            IntegralFunction{false}(dfdp_, nothing)
        end
    end

    ForwardDiff.can_dual(elt) || ForwardDiff.throw_cannot_dual(elt)
    rawp = copy(reinterpret(V, p))

    prob = Integrals.build_problem(cache)
    dp_prob = remake(prob, f = dfdp, p = rawp)
    # the infinity transformation was already applied to f so we don't apply it to dfdp
    dp_cache = init(dp_prob,
        alg;
        sensealg = sensealg,
        do_inf_transformation = Val(false),
        cache.kwargs...)
    dual = Integrals.__solvebp_call(dp_cache, alg, sensealg, domain, rawp; kwargs...)

    res = reinterpret(reshape, DT, dual.u)
    # TODO: if y is a Number/Dual, then return a Number/dual, not an array
    # if y isa AbstractArray
    #     reinterpret(reshape, ForwardDiff.Dual{T, DT, P}, dual.u)
    # else
    #     ForwardDiff.Dual
    # end
    SciMLBase.build_solution(prob, alg, res, dual.resid)
end

duallen(::Type{T}) where {T} = 1
duallen(::T) where {T} = duallen(T)
duallen(::AbstractArray{T}) where {T} = duallen(T)
function duallen(::Type{ForwardDiff.Dual{T, V, P}}) where {T, V, P}
    len = duallen(V)
    return len * (P + 1)
end

replace_dualvaltype(::Type{T}, ::Type{S}) where {T, S} = S
function replace_dualvaltype(::Type{ForwardDiff.Dual{T, V, P}},
        ::Type{S}) where {T, V, P, S}
    return ForwardDiff.Dual{T, replace_dualvaltype(V, S), P}
end

unwrap_dualvaltype(::Type{T}) where {T} = T
function unwrap_dualvaltype(::Type{ForwardDiff.Dual{T, V, P}}) where {T, V, P}
    unwrap_dualvaltype(V)
end
end
