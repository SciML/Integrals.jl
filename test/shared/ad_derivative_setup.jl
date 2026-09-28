using Integrals, Mooncake, Zygote, FiniteDiff, ForwardDiff #, SciMLSensitivity
using Cuba, Cubature
using FastGaussQuadrature
using Test

max_dim_test = 2
max_nout_test = 2

reltol = 1.0e-3
abstol = 1.0e-3

alg_req = Dict(
    QuadratureRule(
        gausslegendre,
        n = 50
    ) => (
        nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false,
    ),
    GaussLegendre(n = 50) => (
        nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false,
    ),
    GaussLegendre(
        n = 50,
        subintervals = 3
    ) => (
        nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false,
    ),
    QuadGKJL() => (
        nout = Inf, allows_batch = true, min_dim = 1, max_dim = 1,
        allows_iip = true,
    ),
    HCubatureJL() => (
        nout = Inf, allows_batch = false, min_dim = 1,
        max_dim = Inf, allows_iip = true,
    ),
    CubatureJLh() => (
        nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true,
    ),
    CubatureJLp() => (
        nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true,
    )
)
# VEGAS() => (nout = 1, allows_batch = true, min_dim = 2, max_dim = Inf,
# allows_iip = true),
# CubaVegas() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
#     allows_iip = true),
# CubaSUAVE() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
#     allows_iip = true),
# CubaDivonne() => (nout = Inf, allows_batch = true, min_dim = 2,
#     max_dim = Inf, allows_iip = true),
# CubaCuhre() => (nout = Inf, allows_batch = true, min_dim = 2, max_dim = Inf,
#     allows_iip = true),

# integrands should have same shape as parameters, independent of dimensionality
integrands = (
    (x, p) -> map(q -> prod(y -> sin(y * q), x), p),
)

# function to turn the output into a scalar / test different tangent types
scalarize_solution = (
    sol -> sin(sum(sol)),
    sol -> sin(sol[1]),
)

# we will be able to use broadcasting for this after https://github.com/FluxML/Zygote.jl/pull/1488
function buffer_copyto!(y, x)
    for (j, i) in zip(eachindex(y), eachindex(x))
        y[j] = x[i]
    end
    return y
end
function f_helper!(f, y, x, p)
    buffer_copyto!(y, f(x, p))
    return
end

# the Zygote implementation is inconsistent about 0-d so we hijack it
struct Scalar{T <: Real} <: Real
    x::T
end
Base.iterate(a::Scalar) = (a.x, nothing)
Base.iterate(::Scalar, _) = nothing
Base.IteratorSize(::Type{Scalar{T}}) where {T} = Base.HasShape{0}()
Base.eltype(::Type{Scalar{T}}) where {T} = T
Base.length(a::Scalar) = 1
Base.size(::Scalar) = ()
Base.:+(a::Scalar, b::Scalar) = Scalar(a.x + b.x)
Base.:*(a::Number, b::Scalar) = a * b.x
Base.:*(a::Scalar, b::Number) = a.x * b
Base.:*(a::Scalar, b::Scalar) = Scalar(a.x * b.x)
Base.zero(a::Scalar) = Scalar(zero(a.x))
Base.map(f, a::Scalar) = map(f, a.x)
(::Type{T})(a::Scalar) where {T <: Real} = T(a.x)
struct ScalarAxes end # the implementation doesn't preserve singleton axes
Base.axes(::Scalar) = ScalarAxes()
Base.iterate(::ScalarAxes) = nothing
Base.reshape(A::AbstractArray, ::ScalarAxes) = Scalar(only(A))

# Scalar struct defined around Real Numbers (test/derivative_tests.jl)
# Mooncake, like Zygote also treats 0-D data wrt to the type of datastructure.
Mooncake.rdata_type(::Type{Scalar{T}}) where {T <: Real} = Mooncake.rdata_type(T)

# here we assume f evaluated at scalar inputs gives a scalar output
# p will be able to be a number  after https://github.com/FluxML/Zygote.jl/pull/1489
# p will be able to be a 0-array after https://github.com/FluxML/Zygote.jl/pull/1491
# p can't be either without both prs
function batch_helper(f, x, p)
    t = f(zero(eltype(x)), zero(eltype(eltype(p))))
    return typeof(t).([f(y, q) for q in p, y in eachslice(x; dims = ndims(x))])
end

function batch_helper!(f, y, x, p)
    buffer_copyto!(y, batch_helper(f, x, p))
    return
end

# helper function / test runner
do_tests = function (; f, scalarize, lb, ub, p, alg, abstol, reltol)
    testf = function (lb, ub, p)
        prob = IntegralProblem(f, (lb, ub), p)
        return scalarize(solve(prob, alg; reltol, abstol))
    end
    testf(lb, ub, p)

    dlb1, dub1,
        dp1 = Zygote.gradient(
        testf, lb, ub, p isa Number && f isa BatchIntegralFunction ? Scalar(p) : p
    )

    f_lb = lb -> testf(lb, ub, p)
    f_ub = ub -> testf(lb, ub, p)

    dlb = lb isa AbstractArray ? :gradient : :derivative
    dub = ub isa AbstractArray ? :gradient : :derivative

    dlb2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dlb))(f_lb, lb)
    dub2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dub))(f_ub, ub)

    if lb isa Number
        @test dlb1 ≈ dlb2 atol = abstol rtol = reltol
        @test dub1 ≈ dub2 atol = abstol rtol = reltol
    else # TODO: implement multivariate limit derivatives in ZygoteExt
        @test_broken dlb1 ≈ dlb2 atol = abstol rtol = reltol
        @test_broken dub1 ≈ dub2 atol = abstol rtol = reltol
    end

    # TODO: implement limit derivatives in ForwardDiffExt
    @test_broken dlb2 ≈ getproperty(ForwardDiff, dlb)(dfdlb, lb) atol = abstol rtol = reltol
    @test_broken dub2 ≈ getproperty(ForwardDiff, dub)(dfdub, ub) atol = abstol rtol = reltol

    f_p = p -> testf(lb, ub, p)

    dp = p isa AbstractArray ? :gradient : :derivative

    dp2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dp))(f_p, p)
    dp3 = getproperty(ForwardDiff, dp)(f_p, p)

    @test dp1 ≈ dp2 atol = abstol rtol = reltol
    @test dp2 ≈ dp3 atol = abstol rtol = reltol

    return
end

# Mooncake Sensealg testing helper function
do_tests_mooncake = function (; f, scalarize, lb, ub, p, alg, abstol, reltol)
    testf = function (lb, ub, p)
        prob = IntegralProblem(f, (lb, ub), p)
        return scalarize(
            solve(
                prob,
                alg;
                reltol,
                abstol,
                sensealg = Integrals.ReCallVJP{Integrals.MooncakeVJP}(Integrals.MooncakeVJP())
            )
        )
    end
    sol_fp = testf(lb, ub, p)

    cache = Mooncake.prepare_gradient_cache(testf, lb, ub, p)
    forwpassval, gradients = Mooncake.value_and_gradient!!(cache, testf, lb, ub, p)

    @test forwpassval == sol_fp

    f_lb = lb -> testf(lb, ub, p)
    f_ub = ub -> testf(lb, ub, p)

    dlb = lb isa AbstractArray ? :gradient : :derivative
    dub = ub isa AbstractArray ? :gradient : :derivative

    dlb2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dlb))(f_lb, lb)
    dub2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dub))(f_ub, ub)

    if lb isa Number
        @test gradients[2] ≈ dlb2 atol = abstol rtol = reltol
        @test gradients[3] ≈ dub2 atol = abstol rtol = reltol
    else # TODO: implement multivariate limit derivatives in MooncakeExt
        @test_broken gradients[2] ≈ dlb2 atol = abstol rtol = reltol
        @test_broken gradients[3] ≈ dub2 atol = abstol rtol = reltol
    end

    f_p = p -> testf(lb, ub, p)
    dp = p isa AbstractArray ? :gradient : :derivative

    dp2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dp))(f_p, p)
    dp3 = getproperty(ForwardDiff, dp)(f_p, p)

    @test dp2 ≈ dp3 atol = abstol rtol = reltol

    # test Mooncake for parameter p
    @test gradients[4] ≈ dp2 atol = abstol rtol = reltol
    @test dp2 ≈ dp3 atol = abstol rtol = reltol

    return
end
