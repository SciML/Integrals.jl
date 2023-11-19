module IntegralsArblibExt

using Arblib
using Integrals

function Integrals.__solvebp_call(prob::IntegralProblem, alg::ArblibJL, sensealg, domain, p;
    reltol = 1e-8, abstol = 1e-8, maxiters = nothing)

    lb, ub = domain
    if lb isa AbstractArray || ub isa AbstractArray
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction

    if isinplace(prob)
        f_ = (y, x; kws...) -> prob.f(y, x, p; kws...)
        val = Arblib.integrate!(f_, lb, ub, atol=abstol, rtol=reltol,
            check_analytic=alg.check_analytic, take_prec=alg.take_prec,
            warn_on_no_convergence=alg.warn_on_no_convergence, opts=alg.opts)
        SciMLBase.build_solution(prob, alg, val, nothing, retcode = ReturnCode.Success)
    else
        f_ = (x; kws...) -> prob.f(x, p; kws...)
        val = Arblib.integrate(f_, lb, ub, atol=abstol, rtol=reltol,
            check_analytic=alg.check_analytic, take_prec=alg.take_prec,
            warn_on_no_convergence=alg.warn_on_no_convergence, opts=alg.opts)
        SciMLBase.build_solution(prob, alg, val, nothing, retcode = ReturnCode.Success)
    end
end

end
