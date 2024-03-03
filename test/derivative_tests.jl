using Integrals, Zygote, FiniteDiff, ForwardDiff#, SciMLSensitivity
using Cuba, Cubature
using FastGaussQuadrature
using Test

max_dim_test = 2
max_nout_test = 2

reltol = 1e-3
abstol = 1e-3

alg_req = Dict(
    QuadratureRule(gausslegendre, n = 50) => (
        nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false),
    GaussLegendre(n = 50) => (nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false),
    QuadGKJL() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = 1,
        allows_iip = true),
    HCubatureJL() => (nout = Inf, allows_batch = false, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    CubatureJLh() => (nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    CubatureJLp() => (nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true))
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
    sol -> sin(sol[1])
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

# here we assume f evaluated at scalar inputs gives a scalar output
# p will be able to be a number  after https://github.com/FluxML/Zygote.jl/pull/1489
# p will be able to be a 0-array after https://github.com/FluxML/Zygote.jl/pull/1491
# p can't be either without both prs
function batch_helper(f, x, p)
    t = f(zero(eltype(x)), zero(eltype(eltype(p))))
    typeof(t).([f(y, q) for q in p, y in eachslice(x; dims = ndims(x))])
end

function batch_helper!(f, y, x, p)
    buffer_copyto!(y, batch_helper(f, x, p))
    return
end

# helper function / test runner
do_tests = function (; f, scalarize, lb, ub, p, alg, abstol, reltol)
    testf = function (lb, ub, p)
        prob = IntegralProblem(f, (lb, ub), p)
        scalarize(solve(prob, alg; reltol, abstol))
    end
    testf(lb, ub, p)

    dlb1, dub1, dp1 = Zygote.gradient(
        testf, lb, ub, p isa Number && f isa BatchIntegralFunction ? Scalar(p) : p)

    f_lb = lb -> testf(lb, ub, p)
    f_ub = ub -> testf(lb, ub, p)

    dlb = lb isa AbstractArray ? :gradient : :derivative
    dub = ub isa AbstractArray ? :gradient : :derivative

    dlb2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dlb))(f_lb, lb)
    dub2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dub))(f_ub, ub)

    if lb isa Number
        @test dlb1≈dlb2 atol=abstol rtol=reltol
        @test dub1≈dub2 atol=abstol rtol=reltol
    else # TODO: implement multivariate limit derivatives in ZygoteExt
        @test_broken dlb1≈dlb2 atol=abstol rtol=reltol
        @test_broken dub1≈dub2 atol=abstol rtol=reltol
    end

    # TODO: implement limit derivatives in ForwardDiffExt
    @test_broken dlb2≈getproperty(ForwardDiff, dlb)(dfdlb, lb) atol=abstol rtol=reltol
    @test_broken dub2≈getproperty(ForwardDiff, dub)(dfdub, ub) atol=abstol rtol=reltol

    f_p = p -> testf(lb, ub, p)

    dp = p isa AbstractArray ? :gradient : :derivative

    dp2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dp))(f_p, p)
    dp3 = getproperty(ForwardDiff, dp)(f_p, p)

    @test dp1≈dp2 atol=abstol rtol=reltol
    @test dp2≈dp3 atol=abstol rtol=reltol

    return
end

### One Dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution)

    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, scalar, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i
    do_tests(; f, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## One-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test

    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, multivariate, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i nout
    do_tests(;
        f, scalarize, lb = 1.0, ub = 3.0, p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test

    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, scalar, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim
    do_tests(; f, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### N-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
    nout in 1:max_nout_test

    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, multivariate, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim nout
    do_tests(; f, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### One Dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution)

    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, scalar, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(1))
    do_tests(; f = fiip, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## One-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test

    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "One-dimensional, multivariate, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i nout
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(nout))
    do_tests(; f = fiip, scalarize, lb = 1.0, ub = 3.0,
        p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test

    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, scalar, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(1))
    do_tests(;
        f = fiip, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### N-dimensional nout iip
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
    nout in 1:max_nout_test

    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, multivariate, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim nout
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(nout))
    do_tests(; f = fiip, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, One Dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution)

    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, scalar, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(; f = bf, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## Batch, One-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test

    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, multivariate, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i nout
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(; f = bf, scalarize, lb = 1.0, ub = 3.0,
        p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test

    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, scalar, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(;
        f = bf, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### Batch, N-dimensional nout
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
    nout in 1:max_nout_test

    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batch, multi-dimensional, multivariate, oop derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim nout
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(; f = bf, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, one-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution)

    req.allows_batch || continue
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, scalar, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i
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

    @info "Batched, one-dimensional, multivariate, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i nout
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(nout, 0))
    do_tests(; f = bfiip, scalarize, lb = 1.0, ub = 3.0,
        p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, N-dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test

    req.allows_batch || continue
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, scalar, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(0))
    do_tests(; f = bfiip, scalarize, lb = ones(dim),
        ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### Batch, N-dimensional nout iip
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
    (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test,
    nout in 1:max_nout_test

    req.allows_batch || continue
    req.allows_iip || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, multivariate, iip derivative test" alg=nameof(typeof(alg)) integrand=j scalarize=i dim nout
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(nout, 0))
    do_tests(; f = bfiip, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol)
end
