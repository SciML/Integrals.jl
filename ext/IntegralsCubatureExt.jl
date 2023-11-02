module IntegralsCubatureExt

using Integrals, Cubature

import Integrals: transformation_if_inf, scale_x, scale_x!
import Cubature: INDIVIDUAL, PAIRED, L1, L2, LINF

CubatureJLh(; error_norm = Cubature.INDIVIDUAL) = CubatureJLh(error_norm)
CubatureJLp(; error_norm = Cubature.INDIVIDUAL) = CubatureJLp(error_norm)

function Integrals.__solvebp_call(prob::IntegralProblem,
    alg::AbstractCubatureJLAlgorithm,
    sensealg, lb, ub, p;
    reltol = 1e-8, abstol = 1e-8,
    maxiters = typemax(Int))
    nout = prob.nout
    if nout == 1
        # the output of prob.f could be either scalar or a vector of length 1, however
        # the behavior of the output of the integration routine is undefined (could differ
        # across algorithms)
        # Cubature will output a real number in when called without nout/fdim
        if prob.batch == 0
            if isinplace(prob)
                dx = zeros(eltype(lb), prob.nout)
                f = (x) -> (prob.f(dx, x, p); dx[1])
            else
                f = (x) -> prob.f(x, p)[1]
            end
            if lb isa Number
                if alg isa CubatureJLh
                    val, err = Cubature.hquadrature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pquadrature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pcubature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            end
        else
            if isinplace(prob)
                f = (x, dx) -> prob.f(dx, x, p)
            else
                f = (x, dx) -> (dx .= prob.f(x, p))
            end
            if lb isa Number
                if alg isa CubatureJLh
                    val, err = Cubature.hquadrature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pquadrature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pcubature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            end
        end
    else
        if prob.batch == 0
            if isinplace(prob)
                f = (x, dx) -> (prob.f(dx, x, p); dx)
            else
                f = (x, dx) -> (dx .= prob.f(x, p))
            end
            if lb isa Number
                if alg isa CubatureJLh
                    val, err = Cubature.hquadrature(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                else
                    val, err = Cubature.pquadrature(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                else
                    val, err = Cubature.pcubature(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                end
            end
        else
            if isinplace(prob)
                f = (x, dx) -> (prob.f(dx, x, p); dx)
            else
                f = (x, dx) -> (dx .= prob.f(x, p))
            end

            if lb isa Number
                if alg isa CubatureJLh
                    val, err = Cubature.hquadrature_v(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                else
                    val, err = Cubature.pquadrature_v(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature_v(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                else
                    val, err = Cubature.pcubature_v(nout, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters,
                        error_norm = alg.error_norm)
                end
            end
        end
    end
    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

end