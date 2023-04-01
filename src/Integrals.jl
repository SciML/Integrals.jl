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

function __solvebp_call(prob::IntegralProblem, alg::QuadGKJL, sensealg, lb, ub, p;
                        reltol = 1e-8, abstol = 1e-8,
                        maxiters = typemax(Int))
    if isinplace(prob) || lb isa AbstractArray || ub isa AbstractArray
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.batch == 0
    @assert prob.nout == 1
    p = p
    f = x -> prob.f(x, p)
    val, err = quadgk(f, lb, ub,
                      rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm)
    SciMLBase.build_solution(prob, QuadGKJL(), val, err, retcode = ReturnCode.Success)
end

function __solvebp_call(prob::IntegralProblem, alg::HCubatureJL, sensealg, lb, ub, p;
                        reltol = 1e-8, abstol = 1e-8,
                        maxiters = typemax(Int))
    p = p

    if isinplace(prob)
        dx = zeros(eltype(lb), prob.nout)
        f = x -> (prob.f(dx, x, prob.p); dx)
    else
        f = x -> prob.f(x, prob.p)
    end
    @assert prob.batch == 0

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

function __solvebp_call(prob::IntegralProblem, alg::VEGAS, sensealg, lb, ub, p;
                        reltol = 1e-8, abstol = 1e-8,
                        maxiters = typemax(Int))
    p = p
    @assert prob.nout == 1
    if prob.batch == 0
        if isinplace(prob)
            dx = zeros(eltype(lb), prob.nout)
            f = x -> (prob.f(dx, x, p); dx[1])
        else
            f = x -> prob.f(x, prob.p)
        end
    else
        if isinplace(prob)
            dx = zeros(eltype(lb), prob.batch)
            f = x -> (prob.f(dx, x', p); dx)
        else
            f = x -> prob.f(x', p)
        end
    end
    ncalls = prob.batch == 0 ? alg.ncalls : prob.batch
    val, err, chi = vegas(f, lb, ub, rtol = reltol, atol = abstol,
                          maxiter = maxiters, nbins = alg.nbins, debug = alg.debug,
                          ncalls = ncalls, batch = prob.batch != 0)
    SciMLBase.build_solution(prob, alg, val, err, chi = chi, retcode = ReturnCode.Success)
end

export QuadGKJL, HCubatureJL, VEGAS, GaussLegendre
end # module
