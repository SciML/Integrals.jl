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

function quadgk_prob_types(f, lb::T, ub::T, p, nrm) where {T}
    DT = float(T)   # we need to be careful to infer the same result as `evalrule`
    RT = Base.promote_op(*, DT, Base.promote_op(f, DT, typeof(p)))    # kernel
    NT = Base.promote_op(nrm, RT)
    return DT, RT, NT
end
function init_cacheval(alg::QuadGKJL, prob::IntegralProblem)
    DT, RT, NT = quadgk_prob_types(prob.f, prob.lb, prob.ub, prob.p, alg.norm)
    return (isconcretetype(RT) ? QuadGK.alloc_segbuf(DT, RT, NT) : nothing)
end
function refresh_cacheval(cacheval, alg::QuadGKJL, prob)
    DT, RT, NT = quadgk_prob_types(prob.f, prob.lb, prob.ub, prob.p, alg.norm)
    isconcretetype(RT) || return nothing
    T = QuadGK.Segment{DT, RT, NT}
    return (cacheval isa Vector{T} ? cacheval : QuadGK.alloc_segbuf(DT, RT, NT))
end

function __solvebp_call(cache::IntegralCache, alg::QuadGKJL, sensealg, lb, ub, p;
    reltol = 1e-8, abstol = 1e-8,
    maxiters = typemax(Int))
    prob = build_problem(cache)
    if isinplace(prob) || lb isa AbstractArray || ub isa AbstractArray
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.batch == 0
    @assert prob.nout == 1

    p = p
    f = x -> prob.f(x, p)
    val, err = quadgk(f, lb, ub, segbuf = cache.cacheval, maxevals = maxiters,
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

is_sampled_problem(prob::IntegralProblem) = prob.f isa AbstractArray
import SciMLBase.IntegralProblem # this is type piracy, and belongs in SciMLBase
function IntegralProblem(y::AbstractArray, lb, ub, args...; kwargs...)
    IntegralProblem{false}(y, lb, ub, args...; kwargs...)
end
function construct_grid(prob, alg, lb, ub, dim)
    x = alg.spec
    if x isa Integer
        @assert length(ub) == length(lb) == 1 "Multidimensional integration is not supported with the Trapezoidal method"
        grid = range(lb[1], ub[1], length=x)
    else 
        grid = x
        @assert ndims(grid) == 1 "Multidimensional integration is not supported with the Trapezoidal method"
    end

    @assert lb[1] ≈ grid[begin] "Lower bound in `IntegralProblem` must coincide with that of the grid"
    @assert ub[1] ≈ grid[end] "Upper bound in `IntegralProblem` must coincide with that of the grid"
    if is_sampled_problem(prob)
        @assert size(prob.f, dim) == length(grid) "Integrand and grid must be of equal length along the integrated dimension"
        @assert axes(prob.f, dim) == axes(grid,1) "Grid and integrand array must use same indexing along integrated dimension" 
    end
    return grid
end

@inline dimension(::Val{D}) where D = D
function __solvebp_call(prob::IntegralProblem, alg::Trapezoidal{S, D}, sensealg, lb, ub, p; kwargs...) where {S,D}
    # since all AbstractRange types are equidistant by design, we can rely on that
    @assert prob.batch == 0
    # using `Val`s for dimensionality is required to make `selectdim` not allocate
    dim = dimension(D) 
    p = p
    if is_sampled_problem(prob)
        @assert alg.spec isa AbstractArray "For pre-sampled problems where the integrand is an array, the integration grid must also be an array."
    end

    grid = construct_grid(prob, alg, lb, ub, dim)
    
    err = Inf64
    if is_sampled_problem(prob)
        # inlining is required in order to not allocate
        @inline function integrand(i) 
            # integrate along dimension `dim`
            selectdim(prob.f, dim, i) 
        end 
    else
        if isinplace(prob)
            y = zeros(eltype(lb), prob.nout)
            integrand = i -> @inbounds (prob.f(y, grid[i], p); y)
        else
            integrand = i -> @inbounds prob.f(grid[i], p)
        end
    end

    firstidx, lastidx = firstindex(grid), lastindex(grid)

    out = integrand(firstidx)

    if isbits(out) 
        # fast path for equidistant grids
        if grid isa AbstractRange 
            dx = grid[begin+1] - grid[begin]
            for i in (firstidx+1):(lastidx-1)
                out += 2*integrand(i)
            end
            out += integrand(lastidx)
            out *= dx/2
        # irregular grids:
        else 
            out *= (grid[firstidx + 1] - grid[firstidx])
            for i in (firstidx+1):(lastidx-1)
                @inbounds out += integrand(i) * (grid[i + 1] - grid[i-1])
            end
            out += integrand(lastidx) * (grid[lastidx] - grid[lastidx-1])
            out /= 2
        end
    else # same, but inplace, broadcasted
        out = copy(out) # to prevent aliasing
        if grid isa AbstractRange 
            dx = grid[begin+1] - grid[begin]
            for i in (firstidx+1):(lastidx-1)
                out .+= 2.0 .* integrand(i)
            end
            out .+= integrand(lastidx)
            out .*= dx/2
        else 
            out .*= (grid[firstidx + 1] - grid[firstidx])
            for i in (firstidx+1):(lastidx-1)
                @inbounds out .+= integrand(i) .* (grid[i + 1] - grid[i-1])
            end
            out .+= integrand(lastidx) .* (grid[lastidx] - grid[lastidx-1])
            out ./= 2
        end
    end

    return SciMLBase.build_solution(prob, alg, out, err, retcode = ReturnCode.Success)
end

export QuadGKJL, HCubatureJL, VEGAS, GaussLegendre, Trapezoidal
end # module
