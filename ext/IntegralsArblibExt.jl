module IntegralsArblibExt

using Arblib
using Integrals
using SciMLLogging: @SciMLMessage

function Integrals.__solvebp_call(
        prob::IntegralProblem, alg::ArblibJL, sensealg, domain, p;
        reltol = 1.0e-8, abstol = 1.0e-8, maxiters = nothing,
        verbose = Integrals.IntegralVerbosity()
    )
    lb_, ub_ = domain
    lb, ub = map(first, domain)
    if !isone(length(lb_)) || !isone(length(ub_))
        error("ArblibJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction

    @SciMLMessage(
        lazy"ArblibJL: starting high-precision integration with reltol=$reltol, abstol=$abstol, check_analytic=$(alg.check_analytic)",
        verbose, :algorithm_selection
    )

    if isinplace(prob)
        @SciMLMessage("Using in-place ball arithmetic evaluation", verbose, :batch_mode)
        res = Acb(0)
        @assert res isa eltype(prob.f.integrand_prototype) "Arblib require inplace prototypes to store Acb elements"
        y_ = similar(prob.f.integrand_prototype)
        f_ = (y, x; kws...) -> (prob.f(y_, x, p; kws...); Arblib.set!(y, only(y_)))
        val = Arblib.integrate!(
            f_, res, lb, ub, atol = abstol, rtol = reltol,
            check_analytic = alg.check_analytic, take_prec = alg.take_prec,
            warn_on_no_convergence = alg.warn_on_no_convergence, opts = alg.opts
        )
    else
        f_ = (x; kws...) -> only(prob.f(x, p; kws...))
        val = Arblib.integrate(
            f_, lb, ub, atol = abstol, rtol = reltol,
            check_analytic = alg.check_analytic, take_prec = alg.take_prec,
            warn_on_no_convergence = alg.warn_on_no_convergence, opts = alg.opts
        )
    end

    @SciMLMessage(
        lazy"ArblibJL converged: val=$val, radius=$(get_radius(val))",
        verbose, :convergence_result
    )

    return SciMLBase.build_solution(prob, alg, val, get_radius(val), retcode = ReturnCode.Success)
end

function get_radius(ball)
    x = abs(midpoint(ball))
    return max(abs(x - abs_ubound(ball)), abs(x - abs_lbound(ball)))
end

end
