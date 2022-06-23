module IntegralsCubature

using Integrals, Cubature

import Integrals: transformation_if_inf, scale_x, scale_x!

abstract type AbstractCubatureJLAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end
struct CubatureJLh <: AbstractCubatureJLAlgorithm end
struct CubatureJLp <: AbstractCubatureJLAlgorithm end

function Integrals.__solvebp_call(
    prob::IntegralProblem,
    alg::AbstractCubatureJLAlgorithm,
    sensealg,
    lb,
    ub,
    p,
    args...;
    reltol = 1e-8,
    abstol = 1e-8,
    maxiters = typemax(Int),
    kwargs...,
)
    prob = transformation_if_inf(prob) #intercept for infinite transformation
    nout = prob.nout
    if nout == 1
        if prob.batch == 0
            if isinplace(prob)
                dx = zeros(prob.nout)
                f = (x) -> (prob.f(dx, x, p); dx[1])
            else
                f = (x) -> prob.f(x, p)[1]
            end
            if lb isa Number
                if alg isa CubatureJLh
                    _val, err = Cubature.hquadrature(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    _val, err = Cubature.pquadrature(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end
                val = prob.f(lb, p) isa Number ? _val : [_val]
            else
                if alg isa CubatureJLh
                    _val, err = Cubature.hcubature(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    _val, err = Cubature.pcubature(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end

                if isinplace(prob) || !isa(prob.f(lb, p), Number)
                    val = [_val]
                else
                    val = _val
                end
            end
        else
            if isinplace(prob)
                f = (x, dx) -> prob.f(dx', x, p)
            elseif lb isa Number
                if prob.f([lb ub], p) isa Vector
                    f = (x, dx) -> (dx .= prob.f(x', p))
                else
                    f = function (x, dx)
                        dx[:] = prob.f(x', p)
                    end
                end
            else
                if prob.f([lb ub], p) isa Vector
                    f = (x, dx) -> (dx .= prob.f(x, p))
                else
                    f = function (x, dx)
                        dx .= prob.f(x, p)[:]
                    end
                end
            end
            if lb isa Number
                if alg isa CubatureJLh
                    _val, err = Cubature.hquadrature_v(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    _val, err = Cubature.pquadrature_v(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end
            else
                if alg isa CubatureJLh
                    _val, err = Cubature.hcubature_v(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    _val, err = Cubature.pcubature_v(
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end
            end
            val = _val isa Number ? [_val] : _val
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
                    val, err = Cubature.hquadrature(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    val, err = Cubature.pquadrature(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    val, err = Cubature.pcubature(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end
            end
        else
            if isinplace(prob)
                f = (x, dx) -> prob.f(dx, x, p)
            else
                if lb isa Number
                    f = (x, dx) -> (dx .= prob.f(x', p))
                else
                    f = (x, dx) -> (dx .= prob.f(x, p))
                end
            end

            if lb isa Number
                if alg isa CubatureJLh
                    val, err = Cubature.hquadrature_v(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    val, err = Cubature.pquadrature_v(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature_v(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                else
                    val, err = Cubature.pcubature_v(
                        nout,
                        f,
                        lb,
                        ub;
                        reltol = reltol,
                        abstol = abstol,
                        maxevals = maxiters,
                    )
                end
            end
        end
    end
    SciMLBase.build_solution(prob, alg, val, err, retcode = :Success)
end

export CubatureJLh, CubatureJLp

end
