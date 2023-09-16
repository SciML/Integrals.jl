module IntegralsCubature

using Integrals, Cubature

import Integrals: transformation_if_inf, scale_x, scale_x!

abstract type AbstractCubatureJLAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end
"""
    CubatureJLh()

Multidimensional h-adaptive integration from Cubature.jl.
`error_norm` specifies the convergence criterion  for vector valued integrands.
Defaults to `Cubature.INDIVIDUAL`, other options are
`Cubature.PAIRED`, `Cubature.L1`, `Cubature.L2`, or `Cubature.LINF`.

## References

@article{genz1980remarks,
title={Remarks on algorithm 006: An adaptive algorithm for numerical integration over an N-dimensional rectangular region},
author={Genz, Alan C and Malik, Aftab Ahmad},
journal={Journal of Computational and Applied mathematics},
volume={6},
number={4},
pages={295--302},
year={1980},
publisher={Elsevier}
}
"""
struct CubatureJLh <: AbstractCubatureJLAlgorithm
    error_norm::Int32
end
CubatureJLh() = CubatureJLh(Cubature.INDIVIDUAL)

"""
    CubatureJLp()

Multidimensional p-adaptive integration from Cubature.jl.
This method is based on repeatedly doubling the degree of the cubature rules,
until convergence is achieved.
The used cubature rule is a tensor product of Clenshaw–Curtis quadrature rules.
`error_norm` specifies the convergence criterion  for vector valued integrands.
Defaults to `Cubature.INDIVIDUAL`, other options are
`Cubature.PAIRED`, `Cubature.L1`, `Cubature.L2`, or `Cubature.LINF`.
"""
struct CubatureJLp <: AbstractCubatureJLAlgorithm
    error_norm::Int32
end
CubatureJLp() = CubatureJLp(Cubature.INDIVIDUAL)

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

export CubatureJLh, CubatureJLp

end
