function evalrule(f, p, lb, ub, nodes, weights)
    scale = map((u, l) -> (u - l) / 2, ub, lb)
    shift = (lb + ub) / 2
    f_ = x -> f(x, p)
    xw = ((map(*, scale, x) + shift, w) for (x, w) in zip(nodes, weights))
    # we are basically computing sum(w .* f.(x))
    # unroll first loop iteration to get right types
    next = iterate(xw)
    next === nothing && throw(ArgumentError("empty quadrature rule"))
    (x0, w0), state = next
    I = w0 * f_(x0)
    next = iterate(xw, state)
    while next !== nothing
        (xi, wi), state = next
        I += wi * f_(xi)
        next = iterate(xw, state)
    end
    return prod(scale) * I
end

function init_cacheval(alg::QuadratureRule, ::IntegralProblem)
    return alg.q(alg.n)
end

function Integrals.__solvebp_call(cache::IntegralCache, alg::QuadratureRule,
        sensealg, domain, p;
        reltol = nothing, abstol = nothing,
        maxiters = nothing)
    prob = build_problem(cache)
    if isinplace(prob)
        error("QuadratureRule does not support inplace integrands.")
    end
    @assert prob.f isa IntegralFunction

    lb, ub = domain
    val = evalrule(cache.f, cache.p, lb, ub, cache.cacheval...)

    err = nothing
    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end
