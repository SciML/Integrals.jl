_oftype(x, y) = oftype(x, y)
_oftype(::SubArray, y) = y

function substitute_bounds(lb, ub)
    mid = (lb + ub) / 2 # floating-point promotion
    if isinf(lb) && isinf(ub)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = flipsign(one(mid), ub)
    elseif isinf(lb)
        lb_sub = flipsign(one(mid), lb)
        ub_sub = zero(one(mid))
    elseif isinf(ub)
        lb_sub = zero(one(mid))
        ub_sub = flipsign(one(mid), ub)
    else
        lb_sub = -one(mid)
        ub_sub = one(mid)
    end
    return lb_sub, ub_sub # unitless
end

function substitute_t(t::Number, lb::Number, ub::Number)
    u = oneunit(eltype(lb))
    # apply correct units
    if isinf(lb) && isinf(ub)
        den = inv(1 - t^2)
        t * den * u, (1 + t^2) * den^2 * u
    elseif isinf(lb)
        den = inv(1 - flipsign(t, lb))
        ub + t * den * u, den^2 * u
    elseif isinf(ub)
        den = inv(1 - flipsign(t, ub))
        lb + t * den * u, den^2 * u
    else
        den = (ub - lb) * oftype(t, 0.5)
        lb + (1 + t) * den, den
    end
end
function substitute_t(t::AbstractVector, lb::AbstractVector, ub::AbstractVector)
    x = similar(t, typeof(one(eltype(t)) * (first(lb) + first(ub))))
    jac = one(eltype(t))
    for i in eachindex(lb)
        x[i], dj = substitute_t(t[i], lb[i], ub[i])
        jac *= dj
    end
    return _oftype(t, x), jac
end

function substitute_f(f::F, t, p, lb, ub) where {F}
    x, jac = substitute_t(t, lb, ub)
    return f(x, p) * jac
end
function substitute_f(f::F, dt, t, p, lb, ub) where {F}
    x, jac = substitute_t(t, lb, ub)
    f(dt, x, p)
    dt .*= jac
    return
end

function substitute_t(t::AbstractVector, lb::Number, ub::Number)
    x = similar(t, typeof(one(eltype(t)) * (lb + ub)))
    jac = similar(x)
    for (i, ti) in enumerate(t)
        x[i], jac[i] = substitute_t(ti, lb, ub)
    end
    return x, jac
end
function substitute_t(t::AbstractArray, lb::AbstractVector, ub::AbstractVector)
    x = similar(t, typeof(one(eltype(t)) * (first(lb) + first(ub))))
    jac = similar(x, size(t, ndims(t)))
    for (i, it) in enumerate(axes(t)[end])
        x[axes(x)[begin:(end - 1)]..., i], jac[i] = substitute_t(
            t[axes(t)[begin:(end - 1)]..., it], lb, ub)
    end
    return x, jac
end

function substitute_batchf(f::F, t, p, lb, ub) where {F}
    x, jac = substitute_t(t, lb, ub)
    r = f(x, p)
    return r .* reshape(jac, ntuple(d -> d == ndims(r) ? length(jac) : 1, ndims(r)))
end
function substitute_batchf(f::F, dt, t, p, lb, ub) where {F}
    x, jac = substitute_t(t, lb, ub)
    f(dt, x, p)
    for (i, j) in zip(axes(dt)[end], jac)
        for idt in CartesianIndices(axes(dt)[begin:(end - 1)])
            dt[idt, i] *= j
        end
    end
    return
end

function transformation_if_inf(prob, ::Val{true})
    lb, ub = promote(prob.domain...)
    f = prob.f
    bounds = map(substitute_bounds, lb, ub)
    lb_sub = lb isa Number ? first(bounds) : map(first, bounds)
    ub_sub = ub isa Number ? last(bounds) : map(last, bounds)
    f_sub = if isinplace(prob)
        if f isa BatchIntegralFunction
            BatchIntegralFunction{true}(
                let f = f.f
                    (dt, t, p) -> substitute_batchf(f, dt, t, p, lb, ub)
                end,
                f.integrand_prototype,
                max_batch = f.max_batch)
        else
            IntegralFunction{true}(let f = f.f
                    (dt, t, p) -> substitute_f(f, dt, t, p, lb, ub)
                end,
                f.integrand_prototype)
        end
    else
        if f isa BatchIntegralFunction
            BatchIntegralFunction{false}(let f = f.f
                    (t, p) -> substitute_batchf(f, t, p, lb, ub)
                end,
                f.integrand_prototype)
        else
            IntegralFunction{false}(let f = f.f
                    (t, p) -> substitute_f(f, t, p, lb, ub)
                end,
                f.integrand_prototype)
        end
    end
    return remake(prob, f = f_sub, domain = (lb_sub, ub_sub))
end

function transformation_if_inf(prob, ::Nothing)
    lb, ub = prob.domain
    if any(isinf, lb) || any(isinf, ub)
        return transformation_if_inf(prob, Val(true))
    else
        return transformation_if_inf(prob, Val(false))
    end
end

function transformation_if_inf(prob, ::Val{false})
    return prob
end

function transformation_if_inf(prob, do_inf_transformation = nothing)
    transformation_if_inf(prob, do_inf_transformation)
end
