function substitute_bounds(lb, ub)
    if isinf(lb) && isinf(ub)
        lb < 0 || error("Positive infinite lower bound not supported.")
        ub > 0 || error("Negative infinite lower bound not supported.")
        lb_sub = -one(lb)
        ub_sub = one(lb)
    elseif isinf(lb)
        lb < 0 || error("Positive infinite lower bound not supported.")
        lb_sub = -one(lb)
        ub_sub = zero(lb)
    elseif isinf(ub)
        ub > 0 || error("Positive infinite lower bound not supported.")
        lb_sub = zero(lb)
        ub_sub = one(lb)
    else
        lb_sub = lb
        ub_sub = ub
    end
    return lb_sub, ub_sub
end
function substitute_f(t, p, f, lb::Number, ub::Number)
    if isinf(lb) && isinf(ub)
        return f(t / (1 - t^2), p) * (1 + t^2) / (1 - t^2)^2
    elseif isinf(lb)
        return f(ub + (t / (1 + t)), p) * 1 / ((1 + t)^2)
    elseif isinf(ub)
        return f(lb + (t / (1 - t)), p) * 1 / ((1 - t)^2)
    else
        return f(t, p)
    end
end
function substitute_f_iip(dt, dy, t, p, f, lb::Number, ub::Number)
    if isinf(lb) && isinf(ub)
        f(dt, t / (1 - t^2), p)
        dt .= dy .* ((1 + t^2) / (1 - t^2)^2)
    elseif isinf(lb)
        return f(ub + (t / (1 + t)), p) * 1 / ((1 + t)^2)
    elseif isinf(ub)
        return f(lb + (t / (1 - t)), p) * 1 / ((1 - t)^2)
    else
        return f(t, p)
    end
end
function substitute_f(t, p, f, lb::AbstractVector, ub::AbstractVector)
    x = similar(t)
    jac_diag = similar(t)
    for i in eachindex(lb)
        if isinf(lb[i]) && isinf(ub[i])
            x[i] = t[i] / (1 - t[i]^2)
            jac_diag[i] = (1 + t[i]^2) / (1 - t[i]^2)^2
        elseif isinf(lb[i])
            x[i] = ub[i] + (t[i] / (1 + t[i]))
            jac_diag[i] = 1 / ((1 + t[i])^2)
        elseif isinf(ub[i])
            x[i] = lb[i] + (t[i] / (1 - t[i]))
            jac_diag[i] = 1 / ((1 - t[i])^2)
        else
            x[i] = t[i]
            jac_diag[i] = one(lb[i])
        end
    end
    f(x, p) * prod(jac_diag)
end
function substitute_f_iip(dt, t, p, f, lb, ub)
    x = similar(t)
    jac_diag = similar(t)
    for i in eachindex(lb)
        if isinf(lb[i]) && isinf(ub[i])
            x[i] = t[i] / (1 - t[i]^2)
            jac_diag[i] = (1 + t[i]^2) / (1 - t[i]^2)^2
        elseif isinf(lb[i])
            x[i] = ub[i] + (t[i] / (1 + t[i]))
            jac_diag[i] = 1 / ((1 + t[i])^2)
        elseif isinf(ub[i])
            x[i] = lb[i] + (t[i] / (1 - t[i]))
            jac_diag[i] = 1 / ((1 - t[i])^2)
        else
            x[i] = t[i]
            jac_diag[i] = one(lb[i])
        end
    end
    f(dt, x, p)
    dt .*= prod(jac_diag)
end

function transformation_if_inf(prob, ::Val{true})
    lb, ub = prob.domain
    f = prob.f
    if lb isa Number
        lb_sub, ub_sub = substitute_bounds(lb, ub)
        # f_sub = (t, p) -> substitute_f_scalar(t, p, f, lb, ub)
    else
        bounds = substitute_bounds.(lb, ub)
        lb_sub = first.(bounds)
        ub_sub = last.(bounds)
        # if isinplace(prob)
        #     f_sub = (dt, t, p) -> substitute_f_vector_iip(dt, t, p, f, lb, ub)
        # else
        #     f_sub = (t, p) -> substitute_f_vector(t, p, f, lb, ub)
        # end
    end
    f_sub = if isinplace(prob)
        if f isa BatchIntegralFunction
            BatchIntegralFunction{true}((dt, t, p) -> substitute_f_iip(dt, t, p, f, lb, ub),
                f.integrand_prototype,
                max_batch = f.max_batch)
        else
            IntegralFunction{true}((dt, t, p) -> substitute_f_iip(dt, t, p, f, lb, ub),
                f.integrand_prototype)
        end
    else
        if f isa BatchIntegralFunction
            BatchIntegralFunction{false}((t, p) -> substitute_f(t, p, f, lb, ub),
                f.integrand_prototype)
        else
            IntegralFunction{false}((t, p) -> substitute_f(t, p, f, lb, ub),
                f.integrand_prototype)
        end
    end
    return remake(prob, f = f_sub, domain = (lb_sub, ub_sub))
end

function transformation_if_inf(prob, ::Nothing)
    lb, ub = prob.domain
    if (lb isa Number && ub isa Number && (ub == Inf || lb == -Inf)) ||
       -Inf in lb || Inf in ub
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
