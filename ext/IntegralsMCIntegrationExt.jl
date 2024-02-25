module IntegralsMCIntegrationExt

using MCIntegration, Integrals

_oftype(::Number, x) = only(x)
_oftype(y, x) = oftype(y, x)

function Integrals.__solvebp_call(prob::IntegralProblem, alg::VEGASMC, sensealg, domain, p;
        reltol = nothing, abstol = nothing, maxiters = 10)
    lb, ub = domain
    mid = (lb + ub) / 2
    tmp = vec(collect(mid))
    var = Continuous(vec([tuple(a, b) for (a, b) in zip(lb, ub)]))

    if prob.f isa BatchIntegralFunction
        error("VEGASMC doesn't support batching. See https://github.com/numericalEFT/MCIntegration.jl/issues/29")
    else
        f0 = if isinplace(prob)
            _f0 = similar(prob.f.integrand_prototype)
            f_ = (x, f, c) -> begin
                n = 0
                for v in x
                    tmp[n += 1] = first(v)
                end
                prob.f(_f0, _oftype(mid, tmp), p)
                f .= vec(_f0)
            end
            _f0
        else
            f_ = (x, c) -> begin
                n = 0
                for v in x
                    tmp[n += 1] = first(v)
                end
                fx = prob.f(_oftype(mid, tmp), p)
                fx isa AbstractArray ? vec(fx) : fx
            end
            prob.f(mid, p)
        end
        dof = ones(Int, length(f0)) # each composite Continuous var gets 1 dof
        res = integrate(f_; var, dof, inplace = isinplace(prob), type = eltype(f0),
            solver = :vegasmc, niter = maxiters, verbose = -2, print = -2, alg.kws...)
        # the package itself is not type-stable
        out::typeof(f0), err::typeof(f0), chi2 = if f0 isa Number
            map(only, (res.mean, res.stdev, res.chi2))
        else
            map(a -> reshape(a, size(f0)), (res.mean, res.stdev, res.chi2))
        end
        chi::typeof(f0) = map(sqrt, chi2)
        SciMLBase.build_solution(prob, alg, out, err, chi = chi,
            retcode = ReturnCode.Success)
    end
end

end
