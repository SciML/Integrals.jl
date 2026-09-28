include(joinpath(@__DIR__, "..", "shared", "ad_derivative_setup.jl"))

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
