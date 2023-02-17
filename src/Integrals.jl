module Integrals

if !isdefined(Base, :get_extension)
    using Requires
end

using Reexport, MonteCarloIntegration, QuadGK, HCubature
@reexport using SciMLBase
using LinearAlgebra
using FastGaussQuadrature

include("init.jl")
include("algorithms.jl")
include("infinity_handling.jl")
include("gaussian_quadrature.jl")

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

function SciMLBase.solve(prob::IntegralProblem; sensealg = ReCallVJP(ZygoteVJP()),
                         do_inf_transformation = nothing, kwargs...)
    if prob.batch != 0
        if prob.nout > 1
            error("Currently no default algorithm for the combination batch > 0 and nout > 1,
                    try the methods in IntegralsCuba or IntegralsCubature")
        end
        return solve(prob, VEGAS(); sensealg = sensealg,
                     do_inf_transformation = do_inf_transformation, kwargs...)
    end

    if prob.nout > 1
        return solve(prob, HCubatureJL(); sensealg = sensealg,
                     do_inf_transformation = do_inf_transformation, kwargs...)
    end

    if prob.lb isa Number
        return solve(prob, QuadGKJL(); sensealg = sensealg,
                     do_inf_transformation = do_inf_transformation, kwargs...)
    elseif length(prob.lb) > 8
        return solve(prob, VEGAS(); sensealg = sensealg,
                     do_inf_transformation = do_inf_transformation, kwargs...)
    else
        return solve(prob, HCubatureJL(); sensealg = sensealg,
                     do_inf_transformation = do_inf_transformation, kwargs...)
    end
end

"""
```julia
solve(prob::IntegralProblem, alg::SciMLBase.AbstractIntegralAlgorithm; kwargs...)
```

## Keyword Arguments

The arguments to `solve` are common across all of the quadrature methods.
These common arguments are:

  - `maxiters` (the maximum number of iterations)
  - `abstol` (absolute tolerance in changes of the objective value)
  - `reltol` (relative tolerance  in changes of the objective value)
"""
function SciMLBase.solve(prob::IntegralProblem,
                         alg::SciMLBase.AbstractIntegralAlgorithm;
                         sensealg = ReCallVJP(ZygoteVJP()),
                         do_inf_transformation = nothing, kwargs...)
    checkkwargs(kwargs...)
    prob = transformation_if_inf(prob, do_inf_transformation)
    __solvebp(prob, alg, sensealg, prob.lb, prob.ub, prob.p; kwargs...)
end
# Throw error if alg is not provided, as defaults are not implemented.
function SciMLBase.solve(::IntegralProblem)
    throw(ArgumentError("""
No integration algorithm `alg` was supplied as the second positional argument.
Reccomended integration algorithms are:
For scalar functions: QuadGKJL()
For ≤ 8 dimensional vector functions: HCubatureJL()
For > 8 dimensional vector functions: MonteCarloIntegration.vegas(f, st, en, kwargs...)
See the docstrings of the different algorithms for more detail.
"""))
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
        dx = zeros(prob.nout)
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
            dx = zeros(prob.nout)
            f = x -> (prob.f(dx, x, p); dx[1])
        else
            f = x -> prob.f(x, prob.p)
        end
    else
        if isinplace(prob)
            dx = zeros(prob.batch)
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

function __solvebp_call(prob::IntegralProblem, alg::GaussLegendre{C}, sensealg, lb, ub, p;
                        reltol = nothing, abstol = nothing, maxiters = nothing) where {C}
    if isinplace(prob) || lb isa AbstractArray || ub isa AbstractArray
        error("GaussLegendre only accepts one-dimensional quadrature problems.")
    end
    @assert prob.batch == 0
    @assert prob.nout == 1
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

export QuadGKJL, HCubatureJL, VEGAS, GaussLegendre
end # module
