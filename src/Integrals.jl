module Integrals

if !isdefined(Base, :get_extension)
    using Requires
end

using Reexport, MonteCarloIntegration, QuadGK, HCubature
@reexport using SciMLBase
using LinearAlgebra

include("common.jl")
include("init.jl")
include("algorithms.jl")
include("infinity_handling.jl")
include("quadrules.jl")
include("sampled.jl")
include("trapezoidal.jl")

abstract type QuadSensitivityAlg end
struct ReCallVJP{V}
    vjp::V
end

abstract type IntegralVJP end
struct ZygoteVJP end
struct ReverseDiffVJP
    compile::Bool
end

function scale_x!(_x, ub, lb, x)
    _x .= (ub .- lb) .* x .+ lb
    _x
end

function scale_x(ub, lb, x)
    (ub .- lb) .* x .+ lb
end

const allowedkeywords = (:maxiters, :abstol, :reltol)
const KWARGERROR_MESSAGE = """
                           Unrecognized keyword arguments found.
                           The only allowed keyword arguments to `solve` are:
                           $allowedkeywords
                           See https://docs.sciml.ai/Integrals/stable/basics/solve/ for more details.
                           """
struct CommonKwargError <: Exception
    kwargs::Any
end
function Base.showerror(io::IO, e::CommonKwargError)
    println(io, KWARGERROR_MESSAGE)
    notin = collect(map(x -> x.first ∉ allowedkeywords, e.kwargs))
    unrecognized = collect(map(x -> x.first, e.kwargs))[notin]
    print(io, "Unrecognized keyword arguments: ")
    printstyled(io, unrecognized; bold = true, color = :red)
    print(io, "\n\n")
end
function checkkwargs(kwargs...)
    if any(x -> x.first ∉ allowedkeywords, kwargs)
        throw(CommonKwargError(kwargs))
    end
    return nothing
end

# Give a layer to intercept with AD
__solvebp(args...; kwargs...) = __solvebp_call(args...; kwargs...)

function quadgk_prob_types(f, lb::T, ub::T, p, nrm) where {T}
    DT = float(T)   # we need to be careful to infer the same result as `evalrule`
    RT = Base.promote_op(*, DT, Base.promote_op(f, DT, typeof(p)))    # kernel
    NT = Base.promote_op(nrm, RT)
    return DT, RT, NT
end
function init_cacheval(alg::QuadGKJL, prob::IntegralProblem)
    lb, ub = prob.domain
    DT, RT, NT = quadgk_prob_types(prob.f, lb, ub, prob.p, alg.norm)
    return (isconcretetype(RT) ? QuadGK.alloc_segbuf(DT, RT, NT) : nothing)
end
function refresh_cacheval(cacheval, alg::QuadGKJL, prob)
    lb, ub = prob.domain
    DT, RT, NT = quadgk_prob_types(prob.f, lb, ub, prob.p, alg.norm)
    isconcretetype(RT) || return nothing
    T = QuadGK.Segment{DT, RT, NT}
    return (cacheval isa Vector{T} ? cacheval : QuadGK.alloc_segbuf(DT, RT, NT))
end

function __solvebp_call(cache::IntegralCache, alg::QuadGKJL, sensealg, domain, p;
        reltol = 1e-8, abstol = 1e-8,
        maxiters = typemax(Int))
    prob = build_problem(cache)
    lb, ub = domain
    if isinplace(prob) || lb isa AbstractArray || ub isa AbstractArray
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.f isa IntegralFunction

    f = x -> prob.f(x, p)
    val, err = quadgk(f, lb, ub, segbuf = cache.cacheval, maxevals = maxiters,
        rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm)
    SciMLBase.build_solution(prob, QuadGKJL(), val, err, retcode = ReturnCode.Success)
end

function __solvebp_call(prob::IntegralProblem, alg::HCubatureJL, sensealg, domain, p;
        reltol = 1e-8, abstol = 1e-8,
        maxiters = typemax(Int))
    lb, ub = domain

    @assert prob.f isa IntegralFunction
    if isinplace(prob)
        # allocate a new output array at each evaluation since HCubature.jl doesn't support
        # inplace ops
        f = x -> (dx = similar(prob.f.integrand_prototype); prob.f(dx, x, prob.p); dx)
    else
        f = x -> prob.f(x, prob.p)
    end

    if lb isa Number
        val, err = hquadrature(f, lb, ub;
            rtol = reltol, atol = abstol,
            maxevals = maxiters, norm = alg.norm, initdiv = alg.initdiv)
    else
        val, err = hcubature(f, lb, ub;
            rtol = reltol, atol = abstol,
            maxevals = maxiters, norm = alg.norm, initdiv = alg.initdiv)
    end
    SciMLBase.build_solution(prob, HCubatureJL(), val, err, retcode = ReturnCode.Success)
end

function __solvebp_call(prob::IntegralProblem, alg::VEGAS, sensealg, domain, p;
        reltol = 1e-8, abstol = 1e-8,
        maxiters = 1000)
    lb, ub = domain
    mid = (lb + ub) / 2
    if prob.f isa BatchIntegralFunction
        if isinplace(prob)
            y = similar(prob.f.integrand_prototype,
                size(prob.f.integrand_prototype)[begin:(end - 1)]...,
                prob.f.max_batch)
            # MonteCarloIntegration v0.0.x passes points as rows of a matrix
            # MonteCarloIntegration v0.1 passes batches as a vector of views of
            # a matrix with points as columns of a matrix
            # see https://github.com/ranjanan/MonteCarloIntegration.jl/issues/16
            # This is an ugly hack that is compatible with both
            f = x -> (prob.f(y, eltype(x) <: SubArray ? parent(first(x)) : x', p); vec(y))
        else
            y = prob.f(mid isa Number ? typeof(mid)[] :
                       Matrix{eltype(mid)}(undef, length(mid), 0),
                p)
            f = x -> prob.f(eltype(x) <: SubArray ? parent(first(x)) : x', p)
        end
    else
        if isinplace(prob)
            @assert prob.f.integrand_prototype isa
                    AbstractArray{<:Real}&&length(prob.f.integrand_prototype) == 1 "VEGAS only supports Float64-valued integrands"
            y = similar(prob.f.integrand_prototype)
            f = x -> (prob.f(y, x, p); only(y))
        else
            y = prob.f(mid, p)
            f = x -> prob.f(x, prob.p)
        end
    end

    if prob.f isa BatchIntegralFunction
        @assert prod(size(y)[begin:(end - 1)]) == 1&&eltype(y) <: Real "VEGAS only supports Float64-valued scalar integrands"
    else
        @assert length(y) == 1&&eltype(y) <: Real "VEGAS only supports Float64-valued scalar integrands"
    end

    ncalls = prob.f isa BatchIntegralFunction ? prob.f.max_batch : alg.ncalls
    out = vegas(f, lb, ub, rtol = reltol, atol = abstol,
        maxiter = maxiters, nbins = alg.nbins, debug = alg.debug,
        ncalls = ncalls, batch = prob.f isa BatchIntegralFunction)
    val, err, chi = out isa Tuple ? out : (out.integral_estimate, out.standard_deviation, out.chi_squared_average)
    SciMLBase.build_solution(prob, alg, val, err, chi = chi, retcode = ReturnCode.Success)
end

export QuadGKJL, HCubatureJL, VEGAS, GaussLegendre, QuadratureRule, TrapezoidalRule
export CubaVegas, CubaSUAVE, CubaDivonne, CubaCuhre
export CubatureJLh, CubatureJLp

end # module
