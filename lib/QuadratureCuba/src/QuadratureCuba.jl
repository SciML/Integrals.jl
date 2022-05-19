module QuadratureCuba

using Quadrature, Cuba
import Quadrature: transformation_if_inf, scale_x, scale_x!

abstract type AbstractCubaAlgorithm <: DiffEqBase.AbstractQuadratureAlgorithm end
struct CubaVegas <: AbstractCubaAlgorithm end
struct CubaSUAVE <: AbstractCubaAlgorithm end
struct CubaDivonne <: AbstractCubaAlgorithm end
struct CubaCuhre <: AbstractCubaAlgorithm end

function Quadrature.__solvebp_call(prob::QuadratureProblem, alg::AbstractCubaAlgorithm, sensealg,
    lb, ub, p, args...;
    reltol=1e-8, abstol=1e-8,
    maxiters=alg isa CubaSUAVE ? 1000000 : typemax(Int),
    kwargs...)
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
                dx .= prob.f(scale_x!(_x, ub, lb, x), p) .* prod((y) -> y[1] - y[2], zip(ub, lb))
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
                        dx .= prob.f(scale_x(ub, lb, x), p)' .* prod((y) -> y[1] - y[2], zip(ub, lb))
                    end
                else
                    f = function (x, dx)
                        dx .= prob.f(scale_x(ub, lb, x), p) .* prod((y) -> y[1] - y[2], zip(ub, lb))
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
                        dx .= prob.f(scale_x(ub, lb, x), p)' .* prod((y) -> y[1] - y[2], zip(ub, lb))
                    end
                else
                    f = function (x, dx)
                        dx .= prob.f(scale_x(ub, lb, x), p) .* prod((y) -> y[1] - y[2], zip(ub, lb))
                    end
                end
            end
        end
    end

    ndim = length(lb)

    nvec = prob.batch == 0 ? 1 : prob.batch

    if alg isa CubaVegas
        out = Cuba.vegas(f, ndim, prob.nout; rtol=reltol,
            atol=abstol, nvec=nvec,
            maxevals=maxiters, kwargs...)
    elseif alg isa CubaSUAVE
        out = Cuba.suave(f, ndim, prob.nout; rtol=reltol,
            atol=abstol, nvec=nvec,
            maxevals=maxiters, kwargs...)
    elseif alg isa CubaDivonne
        out = Cuba.divonne(f, ndim, prob.nout; rtol=reltol,
            atol=abstol, nvec=nvec,
            maxevals=maxiters, kwargs...)
    elseif alg isa CubaCuhre
        out = Cuba.cuhre(f, ndim, prob.nout; rtol=reltol,
            atol=abstol, nvec=nvec,
            maxevals=maxiters, kwargs...)
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

    DiffEqBase.build_solution(prob, alg, val, out.error,
        chi=out.probability, retcode=:Success)
end

export CubaVegas, CubaSUAVE, CubaDivonne, CubaCuhre

end 