
module IntegralsGaussTuranExt
using Base.Threads
using Integrals
if isdefined(Base, :get_extension)
    using Optim
    using TaylorDiff
    using PreallocationTools
else
    using ..Optim
    using ..TaylorDiff
    using ..PreallocationTools
end

########################################################
## Computing Gauss-Turán quadrature rules

function DEFAULT_w(x::T)::T where {T}
    one(T)
end

"""
Cached data for the `GaussTuranLoss!` call.
"""
struct GaussTuranCache{T}
    n::Int
    s::Int
    N::Int
    a::T
    b::T
    ε::T
    rhs_upper::Vector{T}
    rhs_lower::Vector{T}
    M_upper_buffer::LazyBufferCache{typeof(identity)}
    M_lower_buffer::LazyBufferCache{typeof(identity)}
    X_buffer::LazyBufferCache{typeof(identity)}
    A_buffer::LazyBufferCache{typeof(identity)}
    function GaussTuranCache(
            n,
            s,
            N,
            a::T,
            b::T,
            ε::T,
            rhs_upper::Vector{T},
            rhs_lower::Vector{T}
    ) where {T}
        new{T}(
            n,
            s,
            N,
            a,
            b,
            ε,
            rhs_upper,
            rhs_lower,
            LazyBufferCache(),
            LazyBufferCache(),
            LazyBufferCache(),
            LazyBufferCache()
        )
    end
end

"""
Function whose root defines the quadrature rule.
"""
function GaussTuranLoss!(f, ΔX::AbstractVector{T}, cache) where {T}
    (;
    n,
    s,
    N,
    a,
    rhs_upper,
    rhs_lower,
    M_upper_buffer,
    M_lower_buffer,
    A_buffer,
    X_buffer
) = cache
    M_upper = M_upper_buffer[ΔX, (N, N)]
    M_lower = M_lower_buffer[ΔX, (n, N)]
    A = A_buffer[ΔX, N]
    X = X_buffer[ΔX, n]

    # Compute X from ΔX
    cumsum!(X, ΔX)
    X .+= a

    # Evaluating f
    for (i, x) in enumerate(X)
        # Threads.@threads for j = 1:N
        for j in 1:N
            M_upper[j, i:n:N] .= derivatives(x -> f(x, j), x, 1, Val(2s + 1)).value
        end
        Threads.@threads for j in (N + 1):(N + n)
            M_lower[j - N, i:n:N] .= derivatives(x -> f(x, j), x, 1, Val(2s + 1)).value
        end
    end

    # Solving for A
    A .= M_upper \ rhs_upper

    # Computing output
    out = zero(eltype(ΔX))
    for i in eachindex(X)
        out_term = -rhs_lower[i]
        for j in eachindex(A)
            out_term += A[j] * M_lower[i, j]
        end
        out += out_term^2
    end
    sqrt(out)
end

"""
    Callable result object of the Gauss-Turán quadrature rule
    computation algorithm.
"""
struct GaussTuranResult{T, RType, dfType}
    X::Vector{T}
    A::Matrix{T}
    res::RType
    cache::GaussTuranCache
    df::dfType

    function GaussTuranResult(res, cache::GaussTuranCache{T}, df) where {T}
        (; A_buffer, s, n, N, a) = cache
        X = cumsum(res.minimizer) .+ a
        df.f(res.minimizer)
        A = reshape(A_buffer[T[], N], (n, 2s + 1))
        new{T, typeof(res), typeof(df)}(X, A, res, cache, df)
    end
end

"""
    Input: function f(x, d) which gives the dth derivative of f
"""
function (I::GaussTuranResult{T} where {T})(integrand)
    (; X, A, cache) = I
    (; s) = cache
    out = zero(eltype(X))
    for (i, x) in enumerate(X)
        derivs = derivatives(integrand, x, 1.0, Val(2s + 1)).value
        for (m, deriv) in enumerate(derivs)
            out += A[i, m] * deriv
        end
    end
    out
end

"""
    GaussTuran(f, a, b, n, s; w = DEFAULT_w, ε = nothing, X₀ = nothing)

Determine a quadrature rule

I(g) = ∑ₘ∑ᵢ Aₘᵢ * ∂ᵐ⁻¹g(xᵢ)         (m = 1, … 2s + 1, i = 1, …, n)

that gives the precise integral ₐ∫ᵇf(x)dx for given linearly independent functions f₁, f₂, …, f₂₍ₛ₊₁₎ₙ.

Method:

The equations

∑ₘ∑ᵢ Aₘᵢ * ∂ᵐ⁻¹fⱼ(xᵢ) = ₐ∫ᵇw(x)fⱼ(x)dx        j = 1, …, 2(s+1)n

define an overdetermined linear system M(X)A = b in the weights Aₘᵢ for a given X = (x₁, x₂, …, xₙ).
We split the matrix M into a square upper part M_upper of size (2s+1)n x (2s+1)n and a lower part M_lower of size n x (2s+1)n,
and the right hand size b analogously. From this we obtain A = M_upper⁻¹ * b_upper. Then we can asses the correctness of X by comparing
M_lower * A to b_lower, i.e. how well the last n equations holds. This yields the loss function

loss(X) = ||M_lower * A - b_lower||₂ = ||M_lower * M_upper⁻¹ * b_upper - b_lower||₂.

We have the constraints that we want X to be ordered and in the interval (a, b). To achieve this, we formulate the loss
in terms of ΔX = (Δx₁, Δx₂, …, Δxₙ) = (x₁ - a, x₂ - x₁, …, xₙ - xₙ₋₁) on which we set the constraints

ε ≤ Δxᵢ ≤ b - a - 2 * ε     i = 1, …, n
nε ≤ a + ∑ΔX ≤ b - a - ε

where ε is an enforced minimum distance between the nodes. This prevents that consecutive nodes converge towards eachother making
M_upper singular.

## Inputs

  - `f`: Function with signature `f(x::T, j)::T` that returns fⱼ at x
  - `a`: Integration lower bound
  - `b`: Integration upper bound
  - `n`: The number of nodes in the quadrature rule
  - `s`: Determines the highest order derivative required from the functions fⱼ, currently 2(s + 1)

## Keyword Arguments

  - `w`: the integrand weighting function, must have signature w(x::Number)::Number. Defaults to `w(x) = 1`.
  - `ε`: the minimum distance between nodes. Defaults to 1e-3 * (b - a) / (n + 1).
  - `X₀`: The initial guess for the nodes. Defaults to uniformly distributed over (a, b).
  - `integration_kwargs`: The key word arguments passed to `solve` for integrating w * fⱼ
  - `optimization_kwargs`: The key word arguments passed to `Optim.Options` for the minization problem
    for finding X.
"""
function Integrals.GaussTuran(
        f,
        a::T,
        b::T,
        n,
        s;
        w = DEFAULT_w,
        ε = nothing,
        X₀ = nothing,
        integration_kwargs::NamedTuple = (; reltol = 1e-120),
        optimization_options::Optim.Options = Optim.Options()
) where {T <: AbstractFloat}
    # Initial guess
    if isnothing(X₀)
        X₀ = collect(range(a, b, length = n + 2)[2:(end - 1)])
    else
        @assert length(X₀) == n
    end
    ΔX₀ = diff(X₀)
    pushfirst!(ΔX₀, X₀[1] - a)

    # Minimum distance between nodes
    if isnothing(ε)
        ε = 1e-3 * (b - a) / (n + 1)
    else
        @assert 0 < ε ≤ (b - a) / (n + 1)
    end
    ε = T(ε)

    # Integrate w * f
    integrand = (out, x, j) -> out[] = w(x) * f(x, j)
    function integrate(j)
        prob = IntegralProblem{true}(integrand, (a, b), j)
        res = solve(prob, QuadGKJL(); integration_kwargs...)
        res.u[]
    end
    N = (2s + 1) * n
    rhs_upper = [integrate(j) for j in 1:N]
    rhs_lower = [integrate(j) for j in (N + 1):(N + n)]

    # Solving constrained non linear problem for ΔX, see
    # https://julianlsolvers.github.io/Optim.jl/stable/examples/generated/ipnewton_basics/

    # The cache for evaluating GaussTuranLoss
    cache = GaussTuranCache(n, s, N, a, b, ε, rhs_upper, rhs_lower)

    # The function whose root defines the quadrature rule
    # Note: the optimization method requires a Hessian, 
    # which brings the highest order derivative required to 2s + 2
    func(ΔX) = GaussTuranLoss!(f, ΔX, cache)
    df = TwiceDifferentiable(func, ΔX₀; autodiff = :forward)

    # The constraints on ΔX
    ΔX_lb = fill(ε, length(ΔX₀))
    ΔX_ub = fill(b - a - 2 * ε, length(ΔX₀))

    # Defining the variable and constraints nε ≤ a + ∑ΔX ≤ b - a - ε
    sum_variable!(c, ΔX) = (c[1] = a + sum(ΔX); c)
    sum_jacobian!(J, ΔX) = (J[1, :] .= one(eltype(ΔX)); J)
    sum_hessian!(H, ΔX, λ) = nothing
    sum_lb = [n * ε]
    sum_ub = [b - a - ε]
    constraints = TwiceDifferentiableConstraints(
        sum_variable!,
        sum_jacobian!,
        sum_hessian!,
        ΔX_lb,
        ΔX_ub,
        sum_lb,
        sum_ub
    )

    # Solve for the quadrature rule by minimizing the loss function
    res = Optim.optimize(df, constraints, T.(ΔX₀), IPNewton(), optimization_options)

    GaussTuranResult(res, cache, df)
end

end # module IntegralsGaussTuranExt
