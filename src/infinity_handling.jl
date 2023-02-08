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
function substitute_f_scalar(t, p, f, lb, ub)
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
function substitute_f_vector(t, p, f, lb, ub)
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
function transformation_if_inf(prob, ::Val{true})
    lb = prob.lb
    ub = prob.ub
    f = prob.f
    if lb isa Number
        lb_sub, ub_sub = substitute_bounds(lb, ub)
        f_sub = (t, p) -> substitute_f_scalar(t, p, f, lb, ub)
        return remake(prob, f = f_sub, lb = lb_sub, ub = ub_sub)
    else
        bounds = substitute_bounds.(lb, ub)
        lb_sub = first.(bounds)
        ub_sub = last.(bounds)
        f_sub = (t, p) -> substitute_f_vector(t, p, f, lb, ub)
        return remake(prob, f = f_sub, lb = lb_sub, ub = ub_sub)
    end
end

function transformation_if_inf(prob, ::Nothing)
    if (prob.lb isa Number && prob.ub isa Number && (prob.ub == Inf || prob.lb == -Inf)) ||
       -Inf in prob.lb || Inf in prob.ub
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
