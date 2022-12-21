module IntegralsCuba

using Integrals, Cuba
import Integrals: transformation_if_inf, scale_x, scale_x!

abstract type AbstractCubaAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end
"""
    CubaVegas()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Importance sampling is used to reduce variance.
## References
@article{lepage1978new,
  title={A new algorithm for adaptive multidimensional integration},
  author={Lepage, G Peter},
  journal={Journal of Computational Physics},
  volume={27},
  number={2},
  pages={192--203},
  year={1978},
  publisher={Elsevier}
}
"""
struct CubaVegas <: AbstractCubaAlgorithm end
"""
    CubaSUAVE()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Suave stands for subregion-adaptive VEGAS.
Importance sampling and subdivision are thus used to reduce variance.
## References
@article{hahn2005cuba,
  title={Cuba—a library for multidimensional numerical integration},
  author={Hahn, Thomas},
  journal={Computer Physics Communications},
  volume={168},
  number={2},
  pages={78--95},
  year={2005},
  publisher={Elsevier}
}
"""
struct CubaSUAVE <: AbstractCubaAlgorithm end
"""
    CubaDivonne()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Stratified sampling is used to reduce variance.
## References
@article{friedman1981nested,
  title={A nested partitioning procedure for numerical multiple integration},
  author={Friedman, Jerome H and Wright, Margaret H},
  journal={ACM Transactions on Mathematical Software (TOMS)},
  volume={7},
  number={1},
  pages={76--92},
  year={1981},
  publisher={ACM New York, NY, USA}
}
"""
struct CubaDivonne <: AbstractCubaAlgorithm end
"""
    CubaCuhre()

Multidimensional h-adaptive integration from Cuba.jl.
## References
@article{berntsen1991adaptive,
  title={An adaptive algorithm for the approximate calculation of multiple integrals},
  author={Berntsen, Jarle and Espelid, Terje O and Genz, Alan},
  journal={ACM Transactions on Mathematical Software (TOMS)},
  volume={17},
  number={4},
  pages={437--451},
  year={1991},
  publisher={ACM New York, NY, USA}
}
"""
struct CubaCuhre <: AbstractCubaAlgorithm end

function Integrals.__solvebp_call(prob::IntegralProblem, alg::AbstractCubaAlgorithm,
                                  sensealg,
                                  lb, ub, p, args...;
                                  reltol = 1e-8, abstol = 1e-8,
                                  maxiters = alg isa CubaSUAVE ? 1000000 : typemax(Int),
                                  kwargs...)
    @assert maxiters>=1000 "maxiters for $alg should be larger than 1000"
    prob = transformation_if_inf(prob) #intercept for infinite transformation
    p = p
    if lb isa Number && prob.batch == 0
        _x = Float64[lb]
    elseif lb isa Number
        _x = zeros(length(lb), prob.batch)
    elseif prob.batch == 0
        _x = zeros(length(lb))
    else
        _x = zeros(length(lb), prob.batch)
    end
    ub = ub
    lb = lb

    if prob.batch == 0
        if isinplace(prob)
            f = function (x, dx)
                prob.f(dx, scale_x!(_x, ub, lb, x), p)
                dx .*= prod((y) -> y[1] - y[2], zip(ub, lb))
            end
        else
            f = function (x, dx)
                dx .= prob.f(scale_x!(_x, ub, lb, x), p) .*
                      prod((y) -> y[1] - y[2], zip(ub, lb))
            end
        end
    else
        if lb isa Number
            if isinplace(prob)
                f = function (x, dx)
                    #todo check scale_x!
                    prob.f(dx', scale_x!(view(_x, 1:length(x)), ub, lb, x), p)
                    dx .*= prod((y) -> y[1] - y[2], zip(ub, lb))
                end
            else
                if prob.f([lb ub], p) isa Vector
                    f = function (x, dx)
                        dx .= prob.f(scale_x(ub, lb, x), p)' .*
                              prod((y) -> y[1] - y[2], zip(ub, lb))
                    end
                else
                    f = function (x, dx)
                        dx .= prob.f(scale_x(ub, lb, x), p) .*
                              prod((y) -> y[1] - y[2], zip(ub, lb))
                    end
                end
            end
        else
            if isinplace(prob)
                f = function (x, dx)
                    prob.f(dx, scale_x(ub, lb, x), p)
                    dx .*= prod((y) -> y[1] - y[2], zip(ub, lb))
                end
            else
                if prob.f([lb ub], p) isa Vector
                    f = function (x, dx)
                        dx .= prob.f(scale_x(ub, lb, x), p)' .*
                              prod((y) -> y[1] - y[2], zip(ub, lb))
                    end
                else
                    f = function (x, dx)
                        dx .= prob.f(scale_x(ub, lb, x), p) .*
                              prod((y) -> y[1] - y[2], zip(ub, lb))
                    end
                end
            end
        end
    end

    ndim = length(lb)

    nvec = prob.batch == 0 ? 1 : prob.batch

    if alg isa CubaVegas
        out = Cuba.vegas(f, ndim, prob.nout; rtol = reltol,
                         atol = abstol, nvec = nvec,
                         maxevals = maxiters, kwargs...)
    elseif alg isa CubaSUAVE
        out = Cuba.suave(f, ndim, prob.nout; rtol = reltol,
                         atol = abstol, nvec = nvec,
                         maxevals = maxiters, kwargs...)
    elseif alg isa CubaDivonne
        out = Cuba.divonne(f, ndim, prob.nout; rtol = reltol,
                           atol = abstol, nvec = nvec,
                           maxevals = maxiters, kwargs...)
    elseif alg isa CubaCuhre
        out = Cuba.cuhre(f, ndim, prob.nout; rtol = reltol,
                         atol = abstol, nvec = nvec,
                         maxevals = maxiters, kwargs...)
    end

    if isinplace(prob) || prob.batch != 0
        val = out.integral
    else
        if prob.nout == 1 && prob.f(lb, p) isa Number
            val = out.integral[1]
        else
            val = out.integral
        end
    end

    SciMLBase.build_solution(prob, alg, val, out.error,
                             chi = out.probability, retcode = ReturnCode.Success)
end

export CubaVegas, CubaSUAVE, CubaDivonne, CubaCuhre

end
