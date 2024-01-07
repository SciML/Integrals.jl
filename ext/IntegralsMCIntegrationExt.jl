module IntegralsMCIntegrationExt

using MCIntegration, Integrals

function Integrals.__solvebp_call(prob::IntegralProblem, alg::VEGASMC, sensealg, domain, p;
    reltol = nothing, abstol = nothing, maxiters = 1000)
    lb, ub = domain
    mid = vec(collect((lb + ub) / 2))
    vars = Continuous(vec([tuple(a,b) for (a,b) in zip(lb, ub)]))

    if prob.f isa BatchIntegralFunction
        error("VEGASMC doesn't support batching. See https://github.com/numericalEFT/MCIntegration.jl/issues/29")
    else
        if isinplace(prob)
            f0 = similar(prob.f.integrand_prototype)
            f_ = (x, f, c) -> begin
                n = 0
                for v in x
                    mid[n+=1] = first(v)
                end
                prob.f(f0, mid, p)
                f .= vec(f0)
            end
        else
            f0 = prob.f(mid, p)
            f_ = (x, c) -> begin
                n = 0
                for v in x
                    mid[n+=1] = first(v)
                end
                fx = prob.f(mid, p)
                fx isa AbstractArray ? vec(fx) : fx
            end
        end
        dof = ones(Int, length(f0)) # each composite Continuous var gets 1 dof
        res = integrate(f_, inplace=isinplace(prob), var=vars, dof=dof, solver=:vegasmc,
            neval=alg.neval, niter=min(maxiters,alg.niter), block=alg.block, adapt=alg.adapt,
            gamma=alg.gamma, verbose=alg.verbose, debug=alg.debug, type=eltype(f0), print=-2)
        out, err, chi = if f0 isa Number
            map(only, (res.mean, res.stdev, res.chi2))
        else
            map(a -> reshape(a, size(f0)), (res.mean, res.stdev, res.chi2))
        end
        SciMLBase.build_solution(prob, VEGASMC(), out, err, chi=chi, retcode = ReturnCode.Success)
    end
end

end
