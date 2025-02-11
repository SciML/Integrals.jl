module IntegralsMCIntegrationExt

using MCIntegration, Integrals

_oftype(::Number, x) = only(x)
_oftype(y, x) = oftype(y, x)

function Integrals.__solvebp_call(prob::IntegralProblem, alg::VEGASMC, sensealg, domain, p;
        reltol = nothing, abstol = nothing, maxiters = 1000)
    lb, ub = domain
    mid = (lb + ub) / 2
    tmp = vec(collect(mid))
    var = Continuous(vec([tuple(a, b) for (a, b) in zip(lb, ub)]))

    f = prob.f
    if f isa BatchIntegralFunction
        error("VEGASMC doesn't support batching. See https://github.com/numericalEFT/MCIntegration.jl/issues/29")
    else
        prototype = Integrals.get_prototype(prob)
        if isinplace(prob)
            _f = let y = similar(prototype)
                (u, _y, c) -> begin
                    n = 0
                    for v in u
                        tmp[n += 1] = first(v)
                    end
                    f(y, _oftype(mid, tmp), p)
                    _y .= vec(y)
                end
            end
        else
            _f = (u, c) -> begin
                n = 0
                for v in u
                    tmp[n += 1] = first(v)
                end
                y = f(_oftype(mid, tmp), p)
                y isa AbstractArray ? vec(y) : y
            end
        end
        dof = ones(Int, length(prototype)) # each composite Continuous var gets 1 dof
        res = integrate(_f; var, dof, inplace = isinplace(prob), type = eltype(prototype),
            solver = :vegasmc, niter = maxiters, verbose = -2, print = -2, alg.kws...)
        # the package itself is not type-stable
        out::typeof(prototype), err::typeof(prototype), chi2 = if prototype isa Number
            map(only, (res.mean, res.stdev, res.chi2))
        else
            map(a -> reshape(a, size(prototype)), (res.mean, res.stdev, res.chi2))
        end
        chi::typeof(prototype) = map(sqrt, chi2)
        SciMLBase.build_solution(prob, alg, out, err, chi = chi,
            retcode = ReturnCode.Success)
    end
end

end
