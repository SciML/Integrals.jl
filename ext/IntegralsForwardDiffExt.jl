module IntegralsForwardDiffExt
using Integrals
using ForwardDiff
### Forward-Mode AD Intercepts

function Integrals._evaluate!(
        f, y, u,
        p::Union{D, AbstractArray{<:D}}
    ) where {T, V, P, D <: ForwardDiff.Dual{T, V, P}}
    dy = similar(y, replace_dualvaltype(eltype(p), eltype(y)))
    f(dy, u, p)
    return dy
end

# Default to direct AD on solvers
function Integrals.__solvebp(
        cache, alg, sensealg, domain,
        p::Union{D, AbstractArray{<:D}};
        kwargs...
    ) where {T, V, P, D <: ForwardDiff.Dual{T, V, P}}
    return if isinplace(cache.f)
        prototype = cache.f.integrand_prototype
        elt = eltype(prototype)
        ForwardDiff.can_dual(elt) ||
            throw(ArgumentError("ForwardDiff of in-place integrands only supports prototypes with real elements"))
        dprototype = similar(prototype, replace_dualvaltype(D, elt))
        df = if cache.f isa BatchIntegralFunction
            BatchIntegralFunction{true}(cache.f.f, dprototype)
        else
            IntegralFunction{true}(cache.f.f, dprototype)
        end
        prob = Integrals.build_problem(cache)
        dcache = Integrals.IntegralCache(
            cache.iip,
            df,
            domain,
            p,
            cache.prob_kwargs,
            alg,
            sensealg,
            cache.kwargs,
            cache.cacheval
        )
        Integrals.__solvebp_call(dcache, alg, sensealg, domain, p; kwargs...)
    else
        Integrals.__solvebp_call(cache, alg, sensealg, domain, p; kwargs...)
    end
end

# TODO: add the pushforward for derivative w.r.t lb, and ub (and then combinations?)

# Manually split for the pushforward
function Integrals.__solvebp(
        cache, alg::Integrals.AbstractIntegralCExtensionAlgorithm, sensealg, domain,
        p::Union{D, AbstractArray{<:D}};
        kwargs...
    ) where {T, V, P, D <: ForwardDiff.Dual{T, V, P}}

    # we need the output type to avoid perturbation confusion while unwrapping nested duals
    # We compute a vector-valued integral of the primal and dual simultaneously
    if isinplace(cache)
        y = cache.f.integrand_prototype
        elt = eltype(cache.f.integrand_prototype)
        DT = replace_dualvaltype(eltype(p), elt)
        len = duallen(p)
        dual_prototype = similar(
            cache.f.integrand_prototype,
            len,
            size(cache.f.integrand_prototype)...
        )

        dfdp_ = function (out, x, _p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, _p)
            dout = reinterpret(reshape, DT, out)
            cache.f(dout, x, p isa D ? only(dualp) : reshape(dualp, size(p)))
            return
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

        dfdp_ = function (x, _p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, _p)
            ys = cache.f(x, p isa D ? only(dualp) : reshape(dualp, size(p)))
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

    DT <: Real || throw(ArgumentError("differentiating algorithms in C"))
    ForwardDiff.can_dual(elt) || ForwardDiff.throw_cannot_dual(elt)
    rawp = p isa D ? reinterpret(V, [p]) : copy(reinterpret(V, vec(p)))

    prob = Integrals.build_problem(cache)
    dp_prob = remake(prob, f = dfdp, p = rawp)
    dp_cache = init(
        dp_prob,
        alg;
        sensealg = sensealg,
        cache.kwargs...
    )
    dual = solve!(dp_cache)

    res = reinterpret(reshape, DT, dual.u)
    # unwrap the dual when the primal would return a scalar
    out = if (cache.f isa BatchIntegralFunction && y isa AbstractVector) ||
            !(y isa AbstractArray)
        only(res)
    else
        res
    end
    return SciMLBase.build_solution(prob, alg, out, dual.resid)
end

duallen(::Type{T}) where {T} = 1
duallen(::T) where {T} = duallen(T)
duallen(::AbstractArray{T}) where {T} = duallen(T)
function duallen(::Type{ForwardDiff.Dual{T, V, P}}) where {T, V, P}
    len = duallen(V)
    return len * (P + 1)
end

replace_dualvaltype(::Type{T}, ::Type{S}) where {T, S} = S
function replace_dualvaltype(
        ::Type{ForwardDiff.Dual{T, V, P}},
        ::Type{S}
    ) where {T, V, P, S}
    return ForwardDiff.Dual{T, replace_dualvaltype(V, S), P}
end

unwrap_dualvaltype(::Type{T}) where {T} = T
function unwrap_dualvaltype(::Type{ForwardDiff.Dual{T, V, P}}) where {T, V, P}
    return unwrap_dualvaltype(V)
end
end
