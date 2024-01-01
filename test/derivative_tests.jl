using Integrals, Zygote, FiniteDiff, ForwardDiff#, SciMLSensitivity
using Cuba, Cubature
using Test

max_dim_test = 2
max_nout_test = 2

reltol = 1e-3
abstol = 1e-3

alg_req = Dict(
    QuadGKJL() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = 1,
        allows_iip = true),
    HCubatureJL() => (nout = Inf, allows_batch = false, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    # VEGAS() => (nout = 1, allows_batch = true, min_dim = 2, max_dim = Inf,
        # allows_iip = true),
    CubatureJLh() => (nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    CubatureJLp() => (nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    # CubaVegas() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
    #     allows_iip = true),
    CubaSUAVE() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
        allows_iip = true),
    CubaDivonne() => (nout = Inf, allows_batch = true, min_dim = 2,
        max_dim = Inf, allows_iip = true),
    CubaCuhre() => (nout = Inf, allows_batch = true, min_dim = 2, max_dim = Inf,
        allows_iip = true),
)
# helper function / test runner
scalarize_solution = (
    sol -> sin(sum(sol)),
    sol -> sin(sol[1]),
)

do_tests = function (; f, scalarize, lb, ub, p, alg, abstol, reltol)
    testf = function (lb, ub, p)
        prob = IntegralProblem(f, (lb, ub), p)
        scalarize(solve(prob, alg; reltol, abstol))
    end
    testf(lb, ub, p)

    # dlb1, dub1, dp1 = Zygote.gradient(testf, lb, ub, p)

    f_lb = lb -> testf(lb, ub, p)
    f_ub = ub -> testf(lb, ub, p)

    dlb = lb isa AbstractArray ? :gradient : :derivative
    dub = ub isa AbstractArray ? :gradient : :derivative

    dlb2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dlb))(f_lb, lb)
    dub2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dub))(f_ub, ub)

    # if lb isa Number
    #     @test dlb1 ≈ dlb2 atol=abstol rtol=reltol
    #     @test dub1 ≈ dub2 atol=abstol rtol=reltol
    # else # TODO: implement multivariate limit derivatives in ZygoteExt
    #     @test_broken dlb1 ≈ dlb2 atol=abstol rtol=reltol
    #     @test_broken dub1 ≈ dub2 atol=abstol rtol=reltol
    # end

    # TODO: implement limit derivatives in ForwardDiffExt
    @test_broken dlb2 ≈ getproperty(ForwardDiff, dlb)(dfdlb, lb) atol=abstol rtol=reltol
    @test_broken dub2 ≈ getproperty(ForwardDiff, dub)(dfdub, ub) atol=abstol rtol=reltol

    f_p = p -> testf(lb, ub, p)

    dp = p isa AbstractArray ? :gradient : :derivative

    dp2 = getproperty(FiniteDiff, Symbol(:finite_difference_, dp))(f_p, p)
    dp3 = getproperty(ForwardDiff, dp)(f_p, p)

    # @test dp1 ≈ dp2 atol=abstol rtol=reltol
    @test dp2 ≈ dp3 atol=abstol rtol=reltol

    return
end

f_1d_scalar = (x, p) -> sum(q -> sin(q*x), p)
f_1d_nout = (x, p) -> map(q -> q*x, p)
f_nd_scalar = (x, p) -> prod(y -> f_1d_scalar(y, p), x)
f_nd_nout = (x, p) -> mapreduce(y -> f_1d_nout(y, p), +, x)

f_1d_scalar_iip = (y, x, p) -> y .= f_1d_scalar(x, p)
f_1d_nout_iip = (y, x, p) -> y .= f_1d_nout(x, p)
f_nd_scalar_iip = (y, x, p) -> y .= f_nd_scalar(x, p)
f_nd_nout_iip = (y, x, p) -> y .= f_nd_nout(x, p)

bf_helper = (f, x, p) -> begin
    elt = typeof(zero(eltype(x))*zero(eltype(p))) # output type of above functions
    if p isa AbstractArray
        # p and f_*_nout are of size nout
        # this is like a call to stack that should also work for empty arrays
        out = similar(p, elt, size(p)..., size(x, ndims(x)))
        for (v,y) in zip(eachslice(out; dims=ndims(out)), eachslice(x; dims=ndims(x)))
            v .= f(x isa AbstractVector ? only(y) : y, p)
        end
        out
    else
        elt[f(x isa AbstractVector ? only(y) : y, p) for y in eachslice(x; dims=ndims(x))]
    end
end

bf_1d_scalar = (x, p) -> bf_helper(f_1d_scalar, x, p)
bf_1d_nout = (x, p) -> bf_helper(f_1d_nout, x, p)
bf_nd_scalar = (x, p) -> bf_helper(f_nd_scalar, x, p)
bf_nd_nout = (x, p) -> bf_helper(f_nd_nout, x, p)

bf_1d_nout_iip = (y, x, p) -> y .= bf_1d_nout(x, p)
bf_1d_scalar_iip = (y, x, p) -> y .= bf_1d_scalar(x, p)
bf_nd_scalar_iip = (y, x, p) -> y .= bf_nd_scalar(x, p)
bf_nd_nout_iip = (y, x, p) -> y .= bf_nd_nout(x, p)


### One Dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution)
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "One-dimensional, scalar, oop derivative test" alg scalarize=i
    do_tests(; f=f_1d_scalar, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## One-dimensional nout
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "One-dimensional, multivariate, oop derivative test" alg scalarize=i nout
    do_tests(; f=f_1d_nout, scalarize, lb = 1.0, ub = 3.0, p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### N-dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, scalar, oop derivative test" alg scalarize=i dim
    do_tests(; f=f_nd_scalar, scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### N-dimensional nout
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test, nout in 1:max_nout_test
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, multivariate, oop derivative test" alg scalarize=i dim nout
    do_tests(; f=f_nd_nout, scalarize, lb = ones(dim), ub = 3ones(dim), p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### One Dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution)
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "One-dimensional, scalar, iip derivative test" alg scalarize=i
    do_tests(; f=IntegralFunction(f_1d_scalar_iip, zeros(1)), scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## One-dimensional nout
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "One-dimensional, multivariate, iip derivative test" alg scalarize=i nout
    do_tests(; f=IntegralFunction(f_1d_nout_iip, zeros(nout)), scalarize, lb = 1.0, ub = 3.0, p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### N-dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, scalar, iip derivative test" alg scalarize=i dim
    do_tests(; f=IntegralFunction(f_nd_scalar_iip, zeros(1)), scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### N-dimensional nout iip
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test, nout in 1:max_nout_test
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Multi-dimensional, multivariate, iip derivative test" alg scalarize=i dim nout
    do_tests(; f=IntegralFunction(f_nd_nout_iip, zeros(nout)), scalarize, lb = ones(dim), ub = 3ones(dim), p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, One Dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution)
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "Batched, one-dimensional, scalar, oop derivative test" alg scalarize=i
    do_tests(; f=BatchIntegralFunction(bf_1d_scalar), scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## Batch, One-dimensional nout
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "Batched, one-dimensional, multivariate, oop derivative test" alg scalarize=i nout
    do_tests(; f=BatchIntegralFunction(bf_1d_nout), scalarize, lb = 1.0, ub = 3.0, p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, N-dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, scalar, oop derivative test" alg scalarize=i dim
    do_tests(; f=BatchIntegralFunction(bf_nd_scalar), scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### Batch, N-dimensional nout
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test, nout in 1:max_nout_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batch, multi-dimensional, multivariate, oop derivative test" alg scalarize=i dim nout
    do_tests(; f=BatchIntegralFunction(bf_nd_nout), scalarize, lb = ones(dim), ub = 3ones(dim), p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, one-dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution)
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "Batched, one-dimensional, scalar, iip derivative test" alg scalarize=i
    do_tests(; f=BatchIntegralFunction(bf_1d_scalar_iip, zeros(0)), scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
end

## Batch, one-dimensional nout
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), nout in 1:max_nout_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim > 0 || continue

    @info "Batched, one-dimensional, multivariate, iip derivative test" alg scalarize=i nout
    do_tests(; f=BatchIntegralFunction(bf_1d_nout_iip, zeros(nout, 0)), scalarize, lb = 1.0, ub = 3.0, p = [2.0i for i in 1:nout], alg, abstol, reltol)
end

### Batch, N-dimensional
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, scalar, iip derivative test" alg scalarize=i dim
    do_tests(; f=BatchIntegralFunction(bf_nd_scalar_iip, zeros(0)), scalarize, lb = ones(dim), ub = 3ones(dim), p = 2.0, alg, abstol, reltol)
end

### Batch, N-dimensional nout iip
for (alg, req) in pairs(alg_req), (i, scalarize) in enumerate(scalarize_solution), dim in 1:max_dim_test, nout in 1:max_nout_test
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= dim <= req.max_dim || continue

    @info "Batched, multi-dimensional, multivariate, iip derivative test" alg scalarize=i dim nout
    do_tests(; f=BatchIntegralFunction(bf_nd_nout_iip, zeros(nout, 0)), scalarize, lb = ones(dim), ub = 3ones(dim), p = [2.0i for i in 1:nout], alg, abstol, reltol)
end
