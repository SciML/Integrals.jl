include(joinpath(@__DIR__, "..", "shared", "ad_derivative_setup.jl"))

### Batch, One Dimensional
for (alg, req) in pairs(alg_req), (j, f) in enumerate(integrands),
        (i, scalarize) in enumerate(scalarize_solution)
    req.allows_batch || continue
    req.nout > 1 || continue
    req.min_dim <= 1 || continue

    @info "Batched, one-dimensional, scalar, oop derivative test" alg = nameof(typeof(alg)) integrand = j scalarize = i
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(; f = bf, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
    do_tests_mooncake(; f = bf, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
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
    do_tests_mooncake(;
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
    do_tests_mooncake(;
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
    do_tests_mooncake(;
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
    do_tests_mooncake(; f = bfiip, scalarize, lb = 1.0, ub = 3.0, p = 2.0, alg, abstol, reltol)
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
    do_tests_mooncake(;
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
    do_tests_mooncake(;
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
    do_tests_mooncake(;
        f = bfiip, scalarize, lb = ones(dim), ub = 3ones(dim),
        p = [2.0i for i in 1:nout], alg, abstol, reltol
    )
end
