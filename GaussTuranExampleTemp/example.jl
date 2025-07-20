using Optim
using DoubleFloats
using TaylorDiff
using PreallocationTools
using Integrals

# This fix is needed to avoid crashing: https://github.com/JuliaLang/julia/pull/54201
include("linear_algebra_fix.jl")

FT = Double64
n = 5
s = 1
a = FT(0.0)
b = FT(1.0)

"""
    Example functions ϕⱼ from [1]
"""
function f(x::T, j::Int)::T where {T <: Number}
    pow = if j % 2 == 0
        j / 2 - 1
    else
        (j - 1) / 2 - 1 / 3
    end
    x^pow
end

I = Integrals.GaussTuran(
    f,
    a,
    b,
    n,
    s;
    optimization_options = Optim.Options(;
        x_abstol = FT(1e-250),
        g_tol = FT(1e-250),
        show_trace = true,
        show_every = 100,
        iterations = 10_000
    )
)

# Integration error |I(fⱼ) - ₐ∫ᵇfⱼ(x)dx| first N functions fⱼ
max_int_error_upper = maximum(abs(I(x -> f(x, j)) - I.cache.rhs_upper[j])
for j in 1:(I.cache.N))
@show max_int_error_upper
# max_int_error_upper = 2.465190328815662e-32

# Integration error |I(fⱼ) - ₐ∫ᵇfⱼ(x)dx| last n functions fⱼ
max_int_error_lower = maximum(
    abs(I(x -> f(x, j)) - I.cache.rhs_lower[j - I.cache.N])
for
j in (I.cache.N + 1):(I.cache.N + I.cache.n)
)
@show max_int_error_lower
# max_int_error_lower = 9.079315240327527e-30

# Example with eˣ
exp_error = abs(I(Base.exp) - (Base.exp(1) - 1))
@show exp_error;
# exp_error = 6.43662141695295e-18
