module IntegralsFastGaussQuadratureExt
using Integrals
if isdefined(Base, :get_extension)
    import FastGaussQuadrature
    import FastGaussQuadrature: gausslegendre
    # and eventually gausschebyshev, etc.
else
    import ..FastGaussQuadrature
    import ..FastGaussQuadrature: gausslegendre
end
using LinearAlgebra

Integrals.gausslegendre(n) = FastGaussQuadrature.gausslegendre(n)

function gauss_legendre(f::F, p, lb, ub, nodes, weights) where {F}
    scale = (ub - lb) / 2
    shift = (lb + ub) / 2
    I = mapreduce((w, x) -> w * f(scale * x + shift, p), +, weights, nodes)
    return scale * I
end
function composite_gauss_legendre(f::F, p, lb, ub, nodes, weights, subintervals) where {F}
    h = (ub - lb) / subintervals
    I = zero(h)
    for i in 1:subintervals
        _lb = lb + (i - 1) * h
        _ub = _lb + h
        I += gauss_legendre(f, p, _lb, _ub, nodes, weights)
    end
    return I
end

function Integrals.__solvebp_call(prob::IntegralProblem, alg::Integrals.GaussLegendre{C},
        sensealg, domain, p;
        reltol = nothing, abstol = nothing,
        maxiters = nothing) where {C}
    if !all(isone ∘ length, domain)
        error("GaussLegendre only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction
    @assert !isinplace(prob)
    lb, ub = map(only, domain)
    if C
        val = composite_gauss_legendre(prob.f, prob.p, lb, ub,
            alg.nodes, alg.weights, alg.subintervals)
    else
        val = gauss_legendre(prob.f, prob.p, lb, ub,
            alg.nodes, alg.weights)
    end
    err = nothing
    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end
end
