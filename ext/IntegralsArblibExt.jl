module IntegralsArblibExt

using Arblib
using Integrals

function Integrals.__solvebp_call(prob::IntegralProblem, alg::ArblibJL, sensealg, domain, p;
    reltol = 1e-8, abstol = 1e-8, maxiters = nothing)

    lb_, ub_ = domain
    lb, ub = map(first, domain)
    if !isone(length(lb_)) || !isone(length(ub_))
        error("ArblibJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction

    if isinplace(prob)
        y_ = similar(prob.f.integrand_prototype, typeof(Acb(0)))
        f_ = (y, x; kws...) -> Arblib.set!(y, only(prob.f(y_, x, p; kws...)))
        val = Arblib.integrate!(f_, Acb(0), lb, ub, atol=abstol, rtol=reltol,
            check_analytic=alg.check_analytic, take_prec=alg.take_prec,
            warn_on_no_convergence=alg.warn_on_no_convergence, opts=alg.opts)
        SciMLBase.build_solution(prob, alg, val, nothing, retcode = ReturnCode.Success)
    else
        f_ = (x; kws...) -> only(prob.f(x, p; kws...))
        val = Arblib.integrate(f_, lb, ub, atol=abstol, rtol=reltol,
            check_analytic=alg.check_analytic, take_prec=alg.take_prec,
            warn_on_no_convergence=alg.warn_on_no_convergence, opts=alg.opts)
        SciMLBase.build_solution(prob, alg, val, nothing, retcode = ReturnCode.Success)
    end
end

end
