module IntegralsFastTanhSinhQuadratureExt

using Integrals
import FastTanhSinhQuadrature
import FastTanhSinhQuadrature: quad

function Integrals.__solvebp_call(
        prob::IntegralProblem, alg::Integrals.FastTanhSinhQuadratureJL,
        sensealg, domain, p;
        reltol = nothing, abstol = nothing,
        maxiters = nothing
    )
    lb, ub = domain
    f = prob.f

    @assert f isa IntegralFunction "FastTanhSinhQuadratureJL does not support BatchIntegralFunction"
    @assert !isinplace(prob) "FastTanhSinhQuadratureJL does not support in-place integrands"

    # Determine the effective tolerance
    tol = alg.tol
    if reltol !== nothing
        tol = reltol
    end
    max_levels = alg.max_levels

    # Determine dimensionality
    dim = lb isa Number ? 1 : length(lb)

    if dim == 1
        # 1D integration
        _lb = lb isa Number ? lb : only(lb)
        _ub = ub isa Number ? ub : only(ub)
        _f = x -> f.f(x, p)
        val = quad(_f, _lb, _ub; tol = tol, max_levels = max_levels)
    elseif dim == 2
        # 2D integration - use Vectors which quad() converts to SVector internally
        _lb = collect(lb)
        _ub = collect(ub)
        _f = (x, y) -> f.f([x, y], p)
        val = quad(_f, _lb, _ub; tol = tol, max_levels = max_levels)
    elseif dim == 3
        # 3D integration - use Vectors which quad() converts to SVector internally
        _lb = collect(lb)
        _ub = collect(ub)
        _f = (x, y, z) -> f.f([x, y, z], p)
        val = quad(_f, _lb, _ub; tol = tol, max_levels = max_levels)
    else
        error("FastTanhSinhQuadratureJL only supports 1D, 2D, and 3D integration. Got dimension = $dim")
    end

    err = nothing
    return SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

end
