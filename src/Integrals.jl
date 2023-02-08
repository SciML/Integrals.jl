module Integrals

using Reexport, MonteCarloIntegration, QuadGK, HCubature
@reexport using SciMLBase
using Zygote, ForwardDiff, LinearAlgebra

import ChainRulesCore
import ChainRulesCore: NoTangent
import ZygoteRules

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
ChainRulesCore.@non_differentiable checkkwargs(kwargs...)
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

function ChainRulesCore.rrule(::typeof(__solvebp), prob, alg, sensealg, lb, ub, p;
                              kwargs...)
    out = __solvebp_call(prob, alg, sensealg, lb, ub, p; kwargs...)
    function quadrature_adjoint(Δ)
        y = typeof(Δ) <: Array{<:Number, 0} ? Δ[1] : Δ
        if isinplace(prob)
            dx = zeros(prob.nout)
            _f = x -> prob.f(dx, x, p)
            if sensealg.vjp isa ZygoteVJP
                dfdp = function (dx, x, p)
                    _, back = Zygote.pullback(p) do p
                        _dx = Zygote.Buffer(x, prob.nout, size(x, 2))
                        prob.f(_dx, x, p)
                        copy(_dx)
                    end

                    z = zeros(size(x, 2))
                    for idx in 1:size(x, 2)
                        z[1] = 1
                        dx[:, idx] = back(z)[1]
                        z[idx] = 0
                    end
                end
            elseif sensealg.vjp isa ReverseDiffVJP
                error("TODO")
            end
        else
            _f = x -> prob.f(x, p)
            if sensealg.vjp isa ZygoteVJP
                if prob.batch > 0
                    dfdp = function (x, p)
                        _, back = Zygote.pullback(p -> prob.f(x, p), p)

                        out = zeros(length(p), size(x, 2))
                        z = zeros(size(x, 2))
                        for idx in 1:size(x, 2)
                            z[idx] = 1
                            out[:, idx] = back(z)[1]
                            z[idx] = 0
                        end
                        out
                    end
                else
                    dfdp = function (x, p)
                        _, back = Zygote.pullback(p -> prob.f(x, p), p)
                        back(y)[1]
                    end
                end

            elseif sensealg.vjp isa ReverseDiffVJP
                error("TODO")
            end
        end

        dp_prob = remake(prob, f = dfdp, lb = lb, ub = ub, p = p, nout = length(p))

        if p isa Number
            dp = __solvebp_call(dp_prob, alg, sensealg, lb, ub, p; kwargs...)[1]
        else
            dp = __solvebp_call(dp_prob, alg, sensealg, lb, ub, p; kwargs...).u
        end

        if lb isa Number
            dlb = -_f(lb)
            dub = _f(ub)
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), dlb, dub, dp)
        else
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),
                    NoTangent(), dp)
        end
    end
    out, quadrature_adjoint
end

ZygoteRules.@adjoint function ZygoteRules.literal_getproperty(sol::SciMLBase.IntegralSolution,
                                                              ::Val{:u})
    sol.u, Δ -> (SciMLBase.build_solution(sol.prob, sol.alg, Δ, sol.resid),)
end

### Forward-Mode AD Intercepts

# Direct AD on solvers with QuadGK and HCubature
function __solvebp(prob, alg::QuadGKJL, sensealg, lb, ub,
                   p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                   kwargs...) where {T, V, P, N}
    __solvebp_call(prob, alg, sensealg, lb, ub, p; kwargs...)
end

function __solvebp(prob, alg::HCubatureJL, sensealg, lb, ub,
                   p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                   kwargs...) where {T, V, P, N}
    __solvebp_call(prob, alg, sensealg, lb, ub, p; kwargs...)
end

# Manually split for the pushforward
function __solvebp(prob, alg, sensealg, lb, ub,
                   p::AbstractArray{<:ForwardDiff.Dual{T, V, P}, N};
                   kwargs...) where {T, V, P, N}
    primal = __solvebp_call(prob, alg, sensealg, lb, ub, ForwardDiff.value.(p);
                            kwargs...)

    nout = prob.nout * P

    if isinplace(prob)
        dfdp = function (out, x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            if prob.batch > 0
                dx = similar(dualp, prob.nout, size(x, 2))
            else
                dx = similar(dualp, prob.nout)
            end
            prob.f(dx, x, dualp)

            ys = reinterpret(ForwardDiff.Dual{T, V, P}, dx)
            idx = 0
            for y in ys
                for p in ForwardDiff.partials(y)
                    out[idx += 1] = p
                end
            end
            return out
        end
    else
        dfdp = function (x, p)
            dualp = reinterpret(ForwardDiff.Dual{T, V, P}, p)
            ys = prob.f(x, dualp)
            if prob.batch > 0
                out = similar(p, V, nout, size(x, 2))
            else
                out = similar(p, V, nout)
            end

            idx = 0
            for y in ys
                for p in ForwardDiff.partials(y)
                    out[idx += 1] = p
                end
            end

            return out
        end
    end
    rawp = copy(reinterpret(V, p))

    dp_prob = IntegralProblem(dfdp, lb, ub, rawp; nout = nout, batch = prob.batch,
                              kwargs...)
    dual = __solvebp_call(dp_prob, alg, sensealg, lb, ub, rawp; kwargs...)
    res = similar(p, prob.nout)
    partials = reinterpret(typeof(first(res).partials), dual.u)
    for idx in eachindex(res)
        res[idx] = ForwardDiff.Dual{T, V, P}(primal.u[idx], partials[idx])
    end
    if primal.u isa Number
        res = first(res)
    end
    SciMLBase.build_solution(prob, alg, res, primal.resid)
end

export QuadGKJL, HCubatureJL, VEGAS
end # module
