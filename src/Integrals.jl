module Integrals

using Reexport, MonteCarloIntegration, QuadGK, HCubature
@reexport using SciMLBase
using LinearAlgebra
using Random

include("algorithms_meta.jl")
include("common.jl")
include("algorithms.jl")
include("algorithms_sampled.jl")
include("algorithms_extension.jl")
include("infinity_handling.jl")
include("quadrules.jl")
include("sampled.jl")
include("trapezoidal.jl")
include("simpsons.jl")

"""
    QuadSensitivityAlg

Abstract type for quadrature sensitivity algorithms.
"""
abstract type QuadSensitivityAlg end
"""
    ReCallVJP{V}

Wrapper for custom vector-Jacobian product functions in automatic differentiation.

# Fields

  - `vjp::V`: The vector-Jacobian product function
"""
struct ReCallVJP{V}
    vjp::V
end

"""
    IntegralVJP

Abstract type for vector-Jacobian product (VJP) methods used in automatic differentiation
of integrals.
"""
abstract type IntegralVJP end
"""
    ZygoteVJP <: IntegralVJP

Uses Zygote.jl for vector-Jacobian products in automatic differentiation of integrals.
"""
struct ZygoteVJP <: IntegralVJP end
"""
    ReverseDiffVJP <: IntegralVJP

Uses ReverseDiff.jl for vector-Jacobian products in automatic differentiation of integrals.

# Fields

  - `compile::Bool`: Whether to compile the tape for better performance
"""
struct ReverseDiffVJP <: IntegralVJP
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
"""
    CommonKwargError <: Exception

Exception thrown when unrecognized keyword arguments are passed to `solve`.

# Fields

  - `kwargs`: The keyword arguments that were passed
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

# Give a layer for meta algorithms above AD
__solve(args...; kwargs...) = __solvebp(args...; kwargs...)
# Give a layer to intercept with AD
__solvebp(args...; kwargs...) = __solvebp_call(args...; kwargs...)

function init_cacheval(alg::ChangeOfVariables, prob::IntegralProblem)
    f, domain = alg.fu2gv(prob.f, prob.domain)
    cache_alg = init_cacheval(alg.alg, remake(prob, f = f, domain = domain))
    return (alg = cache_alg,)
end

function __solve(cache::IntegralCache, alg::ChangeOfVariables, sensealg, udomain, p;
        kwargs...)
    cacheval = cache.cacheval.alg
    g, vdomain = alg.fu2gv(cache.f, udomain)
    _cache = IntegralCache(Val(isinplace(g)),
        g,
        vdomain,
        p,
        cache.prob_kwargs,
        alg.alg,
        sensealg,
        cache.kwargs,
        cacheval)
    sol = __solve(_cache, alg.alg, sensealg, vdomain, p; kwargs...)
    prob = build_problem(cache)
    return SciMLBase.build_solution(
        prob, alg.alg, sol.u, sol.resid, chi = sol.chi, retcode = sol.retcode, stats = sol.stats)
end

function get_prototype(prob::IntegralProblem)
    f = prob.f
    f.integrand_prototype !== nothing && return f.integrand_prototype
    isinplace(f) && throw(ArgumentError("in-place integral functions require a prototype"))
    lb, ub = prob.domain
    mid = (lb + ub) / 2
    p = prob.p
    if f isa BatchIntegralFunction
        mid isa Number ? f(eltype(mid)[], p) :
        f(Matrix{eltype(mid)}(undef, length(mid), 0), p)
    else
        f(mid, p)
    end
end

function init_cacheval(alg::QuadGKJL, prob::IntegralProblem)
    alg.buffer === nothing && return
    lb, ub = map(first, prob.domain)
    mid = (lb + ub) / 2
    prototype = get_prototype(prob) * mid
    TX = typeof(mid)
    TI = typeof(prototype)
    TE = typeof(alg.norm(prototype))
    QuadGK.alloc_segbuf(TX, TI, TE)
end

function __solvebp_call(cache::IntegralCache, alg::QuadGKJL, sensealg, domain, p;
        reltol = 1e-8, abstol = 1e-8,
        maxiters = typemax(Int))
    prob = build_problem(cache)
    lb, ub = map(first, domain)
    if any(!isone ∘ length, domain)
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end

    f = cache.f
    mid = (lb + ub) / 2
    if f isa BatchIntegralFunction
        if isinplace(f)
            # quadgk only works with vector buffers. If the buffer is an array, we have to
            # turn it into a vector of arrays
            prototype = f.integrand_prototype
            _f = if prototype isa AbstractVector
                BatchIntegrand((y, u) -> f(y, u, p), similar(prototype))
            else
                fsize = size(prototype)[begin:(end - 1)]
                BatchIntegrand{Array{eltype(prototype), length(fsize)}}() do v, u
                    let y = similar(v, eltype(eltype(v)), fsize..., length(v))
                        f(y, u, p)
                        map!(collect, v, eachslice(y; dims = ndims(y)))
                    end
                    return
                end
            end
            val,
            err = quadgk(_f, lb, ub, segbuf = cache.cacheval, maxevals = maxiters,
                rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm)
        else
            prototype = f(typeof(mid)[], p)
            _f = if prototype isa AbstractVector
                BatchIntegrand((y, u) -> y .= f(u, p), prototype)
            else
                BatchIntegrand{Array{eltype(prototype), ndims(prototype) - 1}}() do v, u
                    y = f(u, p)
                    map!(collect, v, eachslice(y; dims = ndims(y)))
                    return
                end
            end
            val,
            err = quadgk(_f, lb, ub, segbuf = cache.cacheval, maxevals = maxiters,
                rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm)
        end
    else
        if isinplace(f)
            result = f.integrand_prototype * mid   # result may have different units than prototype
            _f = (y, u) -> f(y, u, p)
            val,
            err = quadgk!(_f, result, lb, ub, segbuf = cache.cacheval,
                maxevals = maxiters,
                rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm)
        else
            _f = u -> f(u, p)
            val,
            err = quadgk(_f, lb, ub, segbuf = cache.cacheval, maxevals = maxiters,
                rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm)
        end
    end
    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

function init_cacheval(alg::HCubatureJL, prob::IntegralProblem)
    alg.buffer === nothing && return
    prototype = get_prototype(prob)
    lb, ub = map(x -> x isa Number ? tuple(x) : x, prob.domain)
    HCubature.hcubature_buffer(x -> prototype, lb, ub; norm = alg.norm)
end
function __solvebp_call(cache::IntegralCache, alg::HCubatureJL, sensealg, domain, p;
        reltol = 1e-8, abstol = 1e-8,
        maxiters = typemax(Int))
    prob = build_problem(cache)
    lb, ub = domain
    f = cache.f

    @assert f isa IntegralFunction
    if isinplace(f)
        # allocate a new output array at each evaluation since HCubature.jl doesn't support
        # inplace ops
        prototype = f.integrand_prototype
        _f = let f = f.f
            u -> (y = similar(prototype); f(y, u, p); y)
        end
    else
        _f = let f = f.f
            u -> f(u, p)
        end
    end

    val,
    err = if lb isa Number
        hquadrature(_f, lb, ub;
            rtol = reltol, atol = abstol, buffer = cache.cacheval,
            maxevals = maxiters, norm = alg.norm, initdiv = alg.initdiv)
    else
        ret = get_prototype(prob) * (prod(ub - lb) / 2) # this calculation for type stability with vector endpoints
        hcubature(_f, lb, ub;
            rtol = reltol, atol = abstol, buffer = cache.cacheval,
            maxevals = maxiters, norm = alg.norm,
            initdiv = alg.initdiv)::Tuple{typeof(ret), typeof(alg.norm(ret))}
    end
    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

function __solvebp_call(prob::IntegralProblem, alg::VEGAS, sensealg, domain, p;
        reltol = 1e-4, abstol = 1e-4,
        maxiters = 1000)
    lb, ub = domain
    mid = (lb + ub) / 2
    f = prob.f
    alg.seed !== nothing && Random.seed!(alg.seed)
    if f isa BatchIntegralFunction
        # MonteCarloIntegration v0.0.x passes points as rows of a matrix
        # MonteCarloIntegration v0.1 passes batches as a vector of views of
        # a matrix with points as columns of a matrix
        # see https://github.com/ranjanan/MonteCarloIntegration.jl/issues/16
        # This is an ugly hack that is compatible with both
        wrangle = x -> begin
            xx = eltype(x) <: SubArray ? parent(first(x)) : x'
            mid isa Number ? vec(xx) : xx
        end
        if isinplace(prob)
            y = similar(prob.f.integrand_prototype,
                size(prob.f.integrand_prototype)[begin:(end - 1)]...,
                prob.f.max_batch)
            _f = x -> (f(y, wrangle(x), p); vec(y))
        else
            y = mid isa Number ? f(typeof(mid)[], p) :
                f(Matrix{eltype(mid)}(undef, length(mid), 0), p)
            _f = x -> vec(f(wrangle(x), p))
        end
    else
        if isinplace(prob)
            @assert prob.f.integrand_prototype isa
                    AbstractArray{<:Real}&&length(prob.f.integrand_prototype) == 1 "VEGAS only supports Float64-valued integrands"
            y = similar(prob.f.integrand_prototype)
            _f = x -> (prob.f(y, mid isa Number ? only(x) : x, p); only(y))
        else
            y = prob.f(mid, p)
            _f = x -> only(prob.f(mid isa Number ? only(x) : x, prob.p))
        end
    end

    if f isa BatchIntegralFunction
        @assert prod(size(y)[begin:(end - 1)]) == 1&&eltype(y) <: Real "VEGAS only supports Float64-valued scalar integrands"
    else
        @assert length(y) == 1&&eltype(y) <: Real "VEGAS only supports Float64-valued scalar integrands"
    end

    ncalls = alg.ncalls
    out = vegas(_f, lb, ub, rtol = reltol, atol = abstol,
        maxiter = maxiters, nbins = alg.nbins, debug = alg.debug,
        ncalls = ncalls, batch = prob.f isa BatchIntegralFunction)
    val, err,
    chi = out isa Tuple ? out :
          (out.integral_estimate, out.standard_deviation, out.chi_squared_average)
    SciMLBase.build_solution(prob, alg, val, err, chi = chi, retcode = ReturnCode.Success)
end

export QuadGKJL, HCubatureJL, VEGAS, VEGASMC, GaussLegendre, QuadratureRule,
       TrapezoidalRule, SimpsonsRule
export CubaVegas, CubaSUAVE, CubaDivonne, CubaCuhre
export CubatureJLh, CubatureJLp
export ArblibJL

end # module
