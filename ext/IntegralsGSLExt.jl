module IntegralsGSLExt

using GSL
using Integrals
using Integrals: IntegralCache

mutable struct GSLCache{T}
    value::T
end
getvalue(cache::GSLCache) = cache.value

function Integrals.init_cacheval(alg::GSLIntegration{typeof(integration_cquad)}, prob::IntegralProblem)
    ws = integration_cquad_workspace_alloc(alg.kws.wssize)
    gslcache = GSLCache(ws)
    finalizer(integration_cquad_workspace_free∘getvalue, gslcache)
    result = Cdouble[0]
    abserr = Cdouble[0]
    nevals = C_NULL # Csize_t[0]
    return (; gslcache, result, abserr, nevals)
end

function Integrals.__solvebp_call(cache::IntegralCache, alg::GSLIntegration{typeof(integration_cquad)}, sensealg, domain, p;
    reltol = 1e-8, abstol = 1e-8, maxiters = nothing)

    prob = Integrals.build_problem(cache)

    if !all(isone∘length, domain)
        error("GSLIntegration only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction

    f = if isinplace(prob)
        @assert isone(length(prob.f.integrand_prototype)) "GSL only supports scalar, real-valued integrands"
        y = similar(prob.f.integrand_prototype, Cdouble)
        x -> (prob.f(y, x, p); only(y))
    else
        x -> Cdouble(only(prob.f(x, p)))
    end
    # gslf = @gsl_function(f) # broken, see: https://github.com/JuliaMath/GSL.jl/pull/128
    ptr = @cfunction($((x,p) -> f(x)), Cdouble, (Cdouble, Ptr{Cvoid}))
    gslf = gsl_function(Base.unsafe_convert(Ptr{Cvoid},ptr), 0)
    a, b = map(Cdouble∘only, domain)
    (; gslcache, result, abserr, nevals) = cache.cacheval
    integration_cquad(gslf, a, b, abstol, reltol, getvalue(gslcache), result, abserr, nevals)
    return SciMLBase.build_solution(prob, alg, only(result), only(abserr), retcode = ReturnCode.Success)
end

end
