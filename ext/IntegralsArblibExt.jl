module IntegralsArblibExt

using Arblib
using Integrals

function Integrals.__solvebp_call(
        prob::IntegralProblem, alg::ArblibJL, sensealg, domain, p;
        reltol = 1e-8, abstol = 1e-8, maxiters = nothing)
    lb_, ub_ = domain
    lb, ub = map(first, domain)
    if !isone(length(lb_)) || !isone(length(ub_))
        error("ArblibJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction

    if isinplace(prob)
        res = Acb(0)
        @assert res isa eltype(prob.f.integrand_prototype) "Arblib require inplace prototypes to store Acb elements"
        y_ = similar(prob.f.integrand_prototype)
        f_ = (y, x; kws...) -> (prob.f(y_, x, p; kws...); Arblib.set!(y, only(y_)))
        val = Arblib.integrate!(f_, res, lb, ub, atol = abstol, rtol = reltol,
            check_analytic = alg.check_analytic, take_prec = alg.take_prec,
            warn_on_no_convergence = alg.warn_on_no_convergence, opts = alg.opts)
        SciMLBase.build_solution(
            prob, alg, val, get_radius(val), retcode = ReturnCode.Success)
    else
        f_ = (x; kws...) -> only(prob.f(x, p; kws...))
        val = Arblib.integrate(f_, lb, ub, atol = abstol, rtol = reltol,
            check_analytic = alg.check_analytic, take_prec = alg.take_prec,
            warn_on_no_convergence = alg.warn_on_no_convergence, opts = alg.opts)
        SciMLBase.build_solution(
            prob, alg, val, get_radius(val), retcode = ReturnCode.Success)
    end
end

function get_radius(ball)
    x = abs(midpoint(ball))
    return max(abs(x - abs_ubound(ball)), abs(x - abs_lbound(ball)))
end

end
