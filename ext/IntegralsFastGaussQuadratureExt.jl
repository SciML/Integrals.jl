module IntegralsFastGaussQuadratureExt
using Integrals
import FastGaussQuadrature
import FastGaussQuadrature: gausslegendre

using LinearAlgebra

Integrals.gausslegendre(n) = FastGaussQuadrature.gausslegendre(n)

function gauss_legendre(f::F, p, lb, ub, nodes, weights) where {F}
    scale = (ub - lb) / 2
    shift = (lb + ub) / 2
    # TODO reuse Integrals.evalrule instead
    x0, xs = Iterators.peel(nodes)
    w0, ws = Iterators.peel(weights)
    I = w0 * f(scale * x0 + shift, p)
    for (w, x) in zip(ws, xs)
        I += w * f(scale * x + shift, p)
    end
    return scale * I
end
function composite_gauss_legendre(f::F, p, lb, ub, nodes, weights, subintervals) where {F}
    h = (ub - lb) / subintervals
    I = gauss_legendre(f, p, lb, lb + h, nodes, weights)
    for i in 1:(subintervals - 1)
        _lb = lb + i * h
        _ub = _lb + h
        I += gauss_legendre(f, p, _lb, _ub, nodes, weights)
    end
    return I
end

function Integrals.__solvebp_call(
        prob::IntegralProblem, alg::Integrals.GaussLegendre{C},
        sensealg, domain, p;
        reltol = nothing, abstol = nothing,
        maxiters = nothing
    ) where {C}
    if !all(isone ∘ length, domain)
        error("GaussLegendre only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction
    @assert !isinplace(prob)
    lb, ub = map(only, domain)
    if C
        val = composite_gauss_legendre(
            prob.f, prob.p, lb, ub,
            alg.nodes, alg.weights, alg.subintervals
        )
    else
        val = gauss_legendre(
            prob.f, prob.p, lb, ub,
            alg.nodes, alg.weights
        )
    end
    err = nothing
    return SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end
end
