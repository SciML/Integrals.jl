function gauss_legendre(f, p, lb, ub, nodes, weights)
    scale = (ub - lb) / 2
    shift = (lb + ub) / 2
    scaled_f = s -> f(scale * s + shift, p)
    I = dot(weights, @. scaled_f(nodes))
    return scale * I
end
function composite_gauss_legendre(f, p, lb, ub, nodes, weights, subintervals)
    h = (ub - lb) / subintervals
    I = zero(h)
    for i in 1:subintervals
        _lb = lb + (i - 1) * h
        _ub = _lb + h
        I += gauss_legendre(f, p, _lb, _ub, nodes, weights)
    end
    return I
end