module IntegralsHAdaptiveIntegrationExt

using Integrals, HAdaptiveIntegration

using Integrals: HAdaptiveIntegrationJL

function _domain_from_bounds(lb::Number, ub::Number)
    return Segment(lb, ub)
end

function _domain_from_bounds(lb::AbstractVector, ub::AbstractVector)
    d = length(lb)
    if d == 1
        return Segment(lb[1], ub[1])
    elseif d == 2
        return Rectangle(Tuple(lb), Tuple(ub))
    elseif d == 3
        return Cuboid(Tuple(lb), Tuple(ub))
    else
        return Orthotope(Tuple(lb), Tuple(ub))
    end
end

function _domain_from_bounds(lb::Tuple, ub::Tuple)
    d = length(lb)
    if d == 1
        return Segment(lb[1], ub[1])
    elseif d == 2
        return Rectangle(lb, ub)
    elseif d == 3
        return Cuboid(lb, ub)
    else
        return Orthotope(lb, ub)
    end
end

function Integrals.__solvebp_call(
        cache::Integrals.IntegralCache, alg::HAdaptiveIntegrationJL, sensealg, domain, p;
        reltol = 1.0e-8, abstol = 1.0e-8,
        maxiters = typemax(Int)
    )
    prob = Integrals.build_problem(cache)
    f = cache.f

    @assert f isa IntegralFunction "HAdaptiveIntegrationJL does not support BatchIntegralFunction"
    @assert !isinplace(f) "HAdaptiveIntegrationJL does not support in-place integrands"

    # Determine the HAdaptiveIntegration domain
    hadaptive_domain = if domain isa HAdaptiveIntegration.Domain.AbstractDomain
        domain
    else
        lb, ub = domain
        _domain_from_bounds(lb, ub)
    end

    _f = if hadaptive_domain isa Segment
        x -> f(x[1], p)
    else
        x -> f(x, p)
    end

    val, err = HAdaptiveIntegration.integrate(
        _f, hadaptive_domain;
        atol = abstol, rtol = reltol, maxsubdiv = maxiters,
        alg.kws...
    )

    return SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

end
