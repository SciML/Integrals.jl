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

    cache = Mooncake.prepare_gradient_cache(
        testf, lb, ub, p isa Number && f isa BatchIntegralFunction ? Scalar(p) : p
    )
    forwpassval,
        gradients = Mooncake.value_and_gradient!!(
        cache, testf, lb, ub, p isa Number && f isa BatchIntegralFunction ? Scalar(p) : p
    )

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
        @test gradients[2] ≈ dlb2 atol = abstol rtol = reltol
        @test gradients[3] ≈ dub2 atol = abstol rtol = reltol
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

### One Dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution)
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, scalar, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i
    do_tests(; f, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
    do_tests_mooncake(; f, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## One-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, multivariate, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i nout
    do_tests(;
        f, scalarize, lb = 1.0, ub = 3.0, p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
    do_tests_mooncake(;
        f, scalarize, lb = 1.0, ub = 3.0, p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

### N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, scalar, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim
    do_tests(; f, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
    do_tests_mooncake(;
        f, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol
    )
end

### N-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
        nout in 1:max_nout_test
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, multivariate, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim nout
    do_tests(;
        f, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
    do_tests_mooncake(;
        f, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

#### in place IntegralCache, IntegralFunction Tests
### One Dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution)
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, scalar, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(1))
    do_tests(; f = fiip, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
    do_tests_mooncake(;
        f = fiip, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol
    )
end

## One-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, multivariate, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i nout
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(nout))
    do_tests(;
        f = fiip, scalarize, lb = 1.0, ub = 3.0,
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
    do_tests_mooncake(;
        f = fiip, scalarize, lb = 1.0, ub = 3.0,
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

### N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, scalar, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(1))
    do_tests(;
        f = fiip, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol
    )
    do_tests_mooncake(;
        f = fiip, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol
    )
end

### N-dimensional nout iip
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
        nout in 1:max_nout_test
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, multivariate, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim nout
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(nout))
    do_tests(;
        f = fiip, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
    do_tests_mooncake(;
        f = fiip, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

### Batch, One Dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution)
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, scalar, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(; f = bf, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## Batch, One-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, multivariate, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i nout
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(;
        f = bf, scalarize, lb = 1.0, ub = 3.0,
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

### Batch, N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, scalar, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(;
        f = bf, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol
    )
end

### Batch, N-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
        nout in 1:max_nout_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batch, multi-dimensional, multivariate, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim nout
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(;
        f = bf, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

### Batch, one-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution)
    req.allows_batch || continue
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, scalar, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(0))
    do_tests(; f = bfiip, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## Batch, one-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.allows_batch || continue
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, multivariate, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i nout
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(nout, 0))
    do_tests(;
        f = bfiip, scalarize, lb = 1.0, ub = 3.0,
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

### Batch, N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.allows_batch || continue
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, scalar, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(0))
    do_tests(;
        f = bfiip, scalarize, lb = ones(dim),
        ub = 3ones(dim), p = 2.0, alg, abstol, reltol
    )
end

### Batch, N-dimensional nout iip
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
        nout in 1:max_nout_test
    req.allows_batch || continue
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, multivariate, iip derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i dim nout
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(nout, 0))
    do_tests(;
        f = bfiip, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end

@testset "ChangeOfVariables rrules" begin
    alg = QuadGKJL()
    # test a simple u-substitution of x = 2.7u + 1.3
    talg = Integrals.ChangeOfVariables(alg) do f, domain
        if f isa IntegralFunction{false}
            IntegralFunction((x, p) -> f((x - 1.3) / 2.7, p) / 2.7),
                map(x -> 1.3 + 2.7x, domain)
        else
            error("not implemented")
        end
    end

    testf = (
        f, lb, ub, p, alg,
        sensealg,
    ) -> begin
        prob = IntegralProblem(f, (lb, ub), p)
        solve(prob, alg; abstol, reltol, sensealg = sensealg).u
    end
    _testf = (x, p) -> x^2 * p
    lb, ub, p = 1.0, 5.0, 2.0

    @testset "Sensitivity using Zygote" begin
        sensealg = Integrals.ReCallVJP(Integrals.ZygoteVJP())
        sol = Zygote.withgradient(
            (args...) -> testf(_testf, args...), lb, ub, p, alg, sensealg
        )
        tsol = Zygote.withgradient(
            (args...) -> testf(_testf, args...), lb, ub, p, talg, sensealg
        )
        @test sol.val ≈ tsol.val
        # Fundamental theorem of Calculus part 1
        @test sol.grad[1] ≈ tsol.grad[1] ≈ -_testf(lb, p)
        @test sol.grad[2] ≈ tsol.grad[2] ≈ _testf(ub, p)
        # This is to check ∂p
        @test sol.grad[3] ≈ tsol.grad[3]
    end

    @testset "Sensitivity using Mooncake" begin
        sensealg = Integrals.ReCallVJP(Integrals.MooncakeVJP())
        # anonymous function for cache creation and gradient evaluation call must be the same.
        func = (args...) -> testf(_testf, args...)
        cache = Mooncake.prepare_gradient_cache(func, lb, ub, p, alg, sensealg)
        sol = Mooncake.value_and_gradient!!(
            cache, func,
            lb, ub, p, alg, sensealg
        )

        cache = Mooncake.prepare_gradient_cache(func, lb, ub, p, talg, sensealg)
        tsol = Mooncake.value_and_gradient!!(
            cache, func, lb, ub, p, talg, sensealg
        )

        @test sol[1] ≈ tsol[1]
        # Fundamental theorem of Calculus part 1
        @test sol[2][2] ≈ tsol[2][2] ≈ -_testf(lb, p)
        @test sol[2][3] ≈ tsol[2][3] ≈ _testf(ub, p)
        # To check ∂p
        @test sol[2][4] ≈ tsol[2][4]
    end
end

# Test for issue #291: NullParameters should not cause *(Nothing, Float) error
@testset "NullParameters gradient - Issue #291" begin
    # Test that using NullParameters (no explicit p argument) doesn't crash
    # when computing gradients with Zygote
    ps = [1.0f0, 2.0f0]

    # Function that captures parameters in closure (doesn't use p)
    function loss_closure(ps)
        g(x, _) = ps[1] * x + ps[2] * x^2
        y = solve(IntegralProblem(g, (0.0f0, 1.0f0)), HCubatureJL()).u
        abs2(y)
    end

    # This should not throw MethodError: no method matching *(::Nothing, ::Float32)
    @test_nowarn Zygote.gradient(loss_closure, ps)

    # Verify the loss value is computed correctly
    @test loss_closure(ps) ≈ (ps[1] * 0.5f0 + ps[2] / 3.0f0)^2

    # When using explicit p parameter, gradients should work correctly
    function loss_explicit_p(ps)
        g(x, p) = p[1] * x + p[2] * x^2
        y = solve(IntegralProblem(g, (0.0f0, 1.0f0), ps), HCubatureJL()).u
        abs2(y)
    end

    grad_explicit = Zygote.gradient(loss_explicit_p, ps)[1]
    @test grad_explicit !== nothing
    @test length(grad_explicit) == 2

    # Compare with ForwardDiff for correctness
    grad_fd = ForwardDiff.gradient(loss_explicit_p, ps)
    @test grad_explicit ≈ grad_fd rtol = 1.0e-5
end

# DifferentiationInterface extension tests are TODO
# The extension provides the foundation for using ADTypes backends as sensealg
# Full testing requires further integration work with the existing Zygote/Mooncake extensions
