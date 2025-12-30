using Integrals
using Cuba, Cubature, Arblib, FastGaussQuadrature, MCIntegration
using Test

max_dim_test = 2
max_nout_test = 2
reltol = 1e-5
abstol = 1e-5

alg_req = Dict(
    QuadGKJL() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = 1,
        allows_iip = true),
    QuadratureRule(gausslegendre,
        n = 50) => (
        nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false),
    GaussLegendre() => (nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false),
    HCubatureJL() => (nout = Inf, allows_batch = false, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    VEGAS(seed = 109) => (nout = 1, allows_batch = true, min_dim = 2, max_dim = Inf,
        allows_iip = true),
    VEGASMC(seed = 42) => (nout = Inf, allows_batch = false, min_dim = 1, max_dim = Inf,
        allows_iip = true),
    CubatureJLh() => (nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    CubatureJLp() => (nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    CubaVegas() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
        allows_iip = true),
    CubaSUAVE() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
        allows_iip = true),
    CubaDivonne() => (nout = Inf, allows_batch = true, min_dim = 2,
        max_dim = Inf, allows_iip = true),
    CubaCuhre() => (nout = Inf, allows_batch = true, min_dim = 2, max_dim = Inf,
        allows_iip = true),
    ArblibJL() => (
        nout = 1, allows_batch = false, min_dim = 1, max_dim = 1, allows_iip = false)
)

integrands = [
    (x, p) -> 1.0,
    (x, p) -> x isa Number ? cos(x) : prod(cos.(x))
]
iip_integrands = [(dx, x, p) -> (dx .= f(x, p)) for f in integrands]

integrands_v = [(x, p, nout) -> collect(1.0:nout)
                let f = integrands[2]
                    (x, p, nout) -> f(x, p) * collect(1.0:nout)
                end]
iip_integrands_v = [(dx, x, p, nout) -> (dx .= f(x, p, nout)) for f in integrands_v]

exact_sol = [
    (ndim, nout, lb, ub) -> prod(ub - lb),
    (ndim, nout, lb, ub) -> prod(sin.(ub) - sin.(lb))
]

exact_sol_v = [
    (ndim, nout, lb, ub) -> prod(ub - lb) * collect(1.0:nout),
    (ndim, nout, lb, ub) -> exact_sol[2](ndim, nout, lb, ub) * collect(1:nout)
]

batch_f(f) = (pts, p) -> begin
    fevals = zeros(size(pts, ndims(pts)))
    for i in axes(pts, ndims(pts))
        x = pts isa Vector ? pts[i] : pts[:, i]
        fevals[i] = f(x, p)
    end
    fevals
end

batch_iip_f(f) = (fevals, pts, p) -> begin
    ax = axes(pts)
    for i in ax[end]
        x = pts[ax[begin:(end - 1)]..., i]
        fevals[i] = f(x, p)
    end
    nothing
end

batch_f_v(f, nout) = (pts, p) -> begin
    fevals = zeros(nout, size(pts, ndims(pts)))
    for i in axes(pts, ndims(pts))
        x = pts isa Vector ? pts[i] : pts[:, i]
        fevals[:, i] .= f(x, p, nout)
    end
    fevals
end

batch_iip_f_v(f, nout) = (fevals, pts, p) -> begin
    for i in axes(pts, ndims(pts))
        x = pts isa Vector ? pts[i] : pts[:, i]
        fevals[:, i] .= f(x, p, nout)
    end
    return
end

@testset "Standard Single Dimension Integrands" begin
    lb, ub = (1.0, 3.0)
    nout = 1
    dim = 1
    for (alg, req) in pairs(alg_req)
        if req.min_dim > 1
            continue
        end
        for i in 1:length(integrands)
            prob = IntegralProblem(integrands[i], (lb, ub))
            @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
            sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
            @test sol.alg == alg
            @test sol.u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
        end
    end
end

@testset "Standard Integrands" begin
    nout = 1
    for (alg, req) in pairs(alg_req)
        for i in 1:length(integrands)
            for dim in 1:max_dim_test
                lb, ub = (ones(dim), 3ones(dim))
                prob = IntegralProblem(integrands[i], (lb, ub))
                if dim > req.max_dim || dim < req.min_dim
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                @test sol.alg == alg
                @test sol.u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
            end
        end
    end
end

@testset "In-place Standard Integrands" begin
    nout = 1
    for (alg, req) in pairs(alg_req)
        for i in 1:length(iip_integrands)
            for dim in 1:max_dim_test
                lb, ub = (ones(dim), 3ones(dim))
                prob = IntegralProblem(iip_integrands[i], (lb, ub))
                if dim > req.max_dim || dim < req.min_dim || !req.allows_iip
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                @test sol.alg == alg
                if sol.u isa Number
                    @test sol.u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
                else
                    @test sol.u≈[exact_sol[i](dim, nout, lb, ub)] rtol=1e-2
                end
            end
        end
    end
end

@testset "Batched Single Dimension Integrands" begin
    (lb, ub) = (1.0, 3.0)
    (dim, nout) = (1, 1)
    for (alg, req) in pairs(alg_req)
        for i in 1:length(integrands)
            prob = IntegralProblem(BatchIntegralFunction(batch_f(integrands[i])), (lb, ub))
            if req.min_dim > 1 || !req.allows_batch
                continue
            end
            @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
            sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
            @test sol.alg == alg
            @test sol.u[1]≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
        end
    end
end

@testset "Batched Standard Integrands" begin
    nout = 1
    for (alg, req) in pairs(alg_req)
        for i in 1:length(integrands)
            for dim in 1:max_dim_test
                (lb, ub) = (ones(dim), 3ones(dim))
                prob = IntegralProblem(BatchIntegralFunction(batch_f(integrands[i])), (
                    lb, ub))
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                @test sol.alg == alg
                if sol.u isa Number
                    @test sol.u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
                else
                    @test sol.u≈[exact_sol[i](dim, nout, lb, ub)] rtol=1e-2
                end
            end
        end
    end
end

@testset "In-Place Batched Standard Integrands" begin
    nout = 1
    for (alg, req) in pairs(alg_req)
        for i in 1:length(iip_integrands)
            for dim in 1:max_dim_test
                (lb, ub) = (ones(dim), 3ones(dim))
                prob = IntegralProblem(
                    BatchIntegralFunction(batch_iip_f(integrands[i]), zeros(0), max_batch = 1000),
                    (lb, ub))
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch ||
                   !req.allows_iip
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                @test sol.alg == alg
                if sol.u isa Number
                    @test sol.u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
                else
                    @test sol.u≈[exact_sol[i](dim, nout, lb, ub)] rtol=1e-2
                end
            end
        end
    end
end

######## Vector Valued Integrands
@testset "Standard Single Dimension Vector Integrands" begin
    lb, ub = (1.0, 3.0)
    dim = 1
    for (alg, req) in pairs(alg_req)
        for i in 1:length(integrands_v)
            for nout in 1:max_nout_test
                integrand_f = let f = integrands_v[i], nout = nout
                    IntegralFunction((x, p) -> f(x, p, nout), zeros(nout))
                end
                prob = IntegralProblem(integrand_f, (lb, ub))
                if req.min_dim > 1 || req.nout < nout
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                @test sol.alg == alg
                if nout == 1
                    @test sol.u[1]≈exact_sol_v[i](dim, nout, lb, ub)[1] rtol=1e-2
                else
                    @test sol.u≈exact_sol_v[i](dim, nout, lb, ub) rtol=1e-2
                end
            end
        end
    end
end

@testset "Standard Vector Integrands" begin
    for (alg, req) in pairs(alg_req)
        for i in 1:length(integrands_v)
            for dim in 1:max_dim_test
                lb, ub = (ones(dim), 3ones(dim))
                for nout in 1:max_nout_test
                    if dim > req.max_dim || dim < req.min_dim || req.nout < nout
                        continue
                    end
                    integrand_f = let f = integrands_v[i], nout = nout
                        IntegralFunction((x, p) -> f(x, p, nout), zeros(nout))
                    end
                    prob = IntegralProblem(integrand_f, (lb, ub))
                    @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                    sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                    @test sol.alg == alg
                    if nout == 1
                        @test sol.u[1]≈exact_sol_v[i](dim, nout, lb, ub)[1] rtol=1e-2
                    else
                        @test sol.u≈exact_sol_v[i](dim, nout, lb, ub) rtol=1e-2
                    end
                end
            end
        end
    end
end

@testset "In-place Standard Vector Integrands" begin
    for (alg, req) in pairs(alg_req)
        for i in 1:length(iip_integrands_v)
            for dim in 1:max_dim_test
                lb, ub = (ones(dim), 3ones(dim))
                for nout in 1:max_nout_test
                    integrand_f = let f = iip_integrands_v[i]
                        IntegralFunction((dx, x, p) -> f(dx, x, p, nout), zeros(nout))
                    end
                    prob = IntegralProblem(integrand_f, (lb, ub))
                    if dim > req.max_dim || dim < req.min_dim || req.nout < nout ||
                       !req.allows_iip
                        continue
                    end
                    @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                    sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                    @test sol.alg == alg
                    if nout == 1
                        @test sol.u[1]≈exact_sol_v[i](dim, nout, lb, ub)[1] rtol=1e-2
                    else
                        @test sol.u≈exact_sol_v[i](dim, nout, lb, ub) rtol=1e-2
                    end
                end
            end
        end
    end
end

@testset "Batched Single Dimension Vector Integrands" begin
    (lb, ub) = (1.0, 3.0)
    (dim, nout) = (1, 2)
    for (alg, req) in pairs(alg_req)
        for i in 1:length(integrands_v)
            prob = IntegralProblem(BatchIntegralFunction(batch_f_v(integrands_v[i], nout)), (
                lb, ub))
            if req.min_dim > 1 || !req.allows_batch || req.nout < nout
                continue
            end
            @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
            sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
            @test sol.alg == alg
            @test sol.u≈exact_sol_v[i](dim, nout, lb, ub) rtol=1e-2
        end
    end
end

@testset "Batched Standard Vector Integrands" begin
    nout = 2
    for (alg, req) in pairs(alg_req)
        for i in 1:length(integrands_v)
            for dim in 1:max_dim_test
                (lb, ub) = (ones(dim), 3ones(dim))
                prob = IntegralProblem(BatchIntegralFunction(batch_f_v(integrands_v[i], nout)), (
                    lb, ub))
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch ||
                   req.nout < nout
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                @test sol.alg == alg
                @test sol.u≈exact_sol_v[i](dim, nout, lb, ub) rtol=1e-2
            end
        end
    end
end

@testset "In-Place Batched Standard Vector Integrands" begin
    nout = 2
    for (alg, req) in pairs(alg_req)
        for i in 1:length(iip_integrands_v)
            for dim in 1:max_dim_test
                (lb, ub) = (ones(dim), 3ones(dim))
                prob = IntegralProblem(
                    BatchIntegralFunction(batch_iip_f_v(integrands_v[i], nout), zeros(nout, 0)),
                    (lb, ub))
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch ||
                   !req.allows_iip || req.nout < nout
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = @inferred solve(prob, alg, reltol = reltol, abstol = abstol)
                @test sol.alg == alg
                @test sol.u≈exact_sol_v[i](dim, nout, lb, ub) rtol=1e-2
            end
        end
    end
end

@testset "Allowed keyword test" begin
    f(u, p) = sum(sin.(u))
    prob = IntegralProblem(f, ones(3), 3ones(3))
    @test_throws Integrals.CommonKwargError((:relztol => 1e-3, :abstol => 1e-3)) solve(
        prob,
        HCubatureJL();
        relztol = 1e-3,
        abstol = 1e-3)
end

@testset "Caching interface" begin
    lb, ub = (1.0, 3.0)
    p = NaN # the integrands don't actually use this
    nout = 1
    dim = 1
    for (alg, req) in pairs(alg_req)
        if req.min_dim > 1
            continue
        end
        for i in 1:length(integrands)
            prob = IntegralProblem(integrands[i], (lb, ub), p)
            cache = init(prob, alg, reltol = reltol, abstol = abstol)
            @test solve!(cache).u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
            cache.lb = lb = 0.5
            @test solve!(cache).u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
            cache.ub = ub = 3.5
            @test solve!(cache).u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
            cache.p = Inf
            @test solve!(cache).u≈exact_sol[i](dim, nout, lb, ub) rtol=1e-2
        end
    end
end

@testset "issue242" begin
    f242(x, p) = p / (x^2 + p^2)
    domain242 = (-1, 1)
    p242 = 1e-3
    for abstol in [1e-2, 1e-4, 1e-6, 1e-8]
        @test solve(IntegralProblem(f242, domain242, p242; abstol), QuadGKJL()).u ==
              solve(IntegralProblem(f242, domain242, p242), QuadGKJL(); abstol).u
    end
end

@testset "Numeric Type Interface" begin
    @testset "BigFloat support" begin
        # Test QuadGKJL with BigFloat bounds
        f_bf = (x, p) -> sin(x)
        lb_bf = BigFloat("0.0")
        ub_bf = BigFloat("1.0")
        prob_bf = IntegralProblem(f_bf, (lb_bf, ub_bf))
        sol_bf = solve(prob_bf, QuadGKJL())
        expected_bf = 1 - cos(BigFloat(1))
        @test sol_bf.u isa BigFloat
        @test sol_bf.u≈expected_bf rtol=1e-10

        # Test HCubatureJL with BigFloat scalar bounds
        sol_hc_bf = solve(prob_bf, HCubatureJL())
        @test sol_hc_bf.u isa BigFloat
        @test sol_hc_bf.u≈expected_bf rtol=1e-10

        # Test HCubatureJL with BigFloat vector bounds
        f_bf_2d = (x, p) -> sin(x[1]) * cos(x[2])
        lb_bf_2d = BigFloat[0, 0]
        ub_bf_2d = BigFloat[1, 1]
        prob_bf_2d = IntegralProblem(f_bf_2d, (lb_bf_2d, ub_bf_2d))
        sol_bf_2d = solve(prob_bf_2d, HCubatureJL())
        @test sol_bf_2d.u isa BigFloat

        # Test SampledIntegralProblem with BigFloat
        x_bf = range(BigFloat("0.0"), BigFloat("1.0"), length = 11)
        y_bf = sin.(x_bf)
        prob_sampled_bf = SampledIntegralProblem(y_bf, x_bf)
        sol_sampled_bf = solve(prob_sampled_bf, TrapezoidalRule())
        @test sol_sampled_bf.u isa BigFloat
        @test eltype(x_bf) == BigFloat
        @test eltype(y_bf) == BigFloat

        # Test SimpsonsRule with BigFloat
        sol_simpson_bf = solve(prob_sampled_bf, SimpsonsRule())
        @test sol_simpson_bf.u isa BigFloat

        # Test BigFloat matrix for vector-valued sampled integration
        y_bf_matrix = BigFloat[sin(xi) * j for j in 1:3, xi in x_bf]
        prob_sampled_bf_2d = SampledIntegralProblem(y_bf_matrix, x_bf; dim = 2)
        sol_sampled_bf_2d = solve(prob_sampled_bf_2d, TrapezoidalRule())
        @test eltype(sol_sampled_bf_2d.u) == BigFloat
    end

    @testset "Complex number support" begin
        # Test SampledIntegralProblem with Complex numbers
        x_c = range(0.0, 1.0, length = 11)
        y_c = complex.(sin.(x_c), cos.(x_c))
        prob_c = SampledIntegralProblem(y_c, x_c)
        sol_c = solve(prob_c, TrapezoidalRule())
        @test sol_c.u isa Complex
        @test real(sol_c.u) > 0
        @test imag(sol_c.u) > 0

        # Test QuadGKJL with complex-valued integrand
        f_c = (x, p) -> exp(im * x)
        prob_quadgk_c = IntegralProblem(f_c, (0.0, 1.0))
        sol_quadgk_c = solve(prob_quadgk_c, QuadGKJL())
        expected_c = (exp(im) - 1) / im
        @test sol_quadgk_c.u isa Complex
        @test sol_quadgk_c.u≈expected_c rtol=1e-8
    end

    @testset "Float32 type preservation" begin
        # Test SampledIntegralProblem preserves Float32
        x_f32 = range(0.0f0, 1.0f0, length = 11)
        y_f32 = sin.(x_f32)
        prob_f32 = SampledIntegralProblem(y_f32, x_f32)
        sol_f32 = solve(prob_f32, TrapezoidalRule())
        @test sol_f32.u isa Float32

        # Test Float32 matrix
        y_f32_matrix = Float32[sin(xi) * j for j in 1:3, xi in x_f32]
        prob_f32_2d = SampledIntegralProblem(y_f32_matrix, x_f32; dim = 2)
        sol_f32_2d = solve(prob_f32_2d, TrapezoidalRule())
        @test eltype(sol_f32_2d.u) == Float32
    end

    @testset "SubArray support" begin
        # Test that views/SubArrays work correctly
        x_full = range(0.0, 1.0, length = 11)
        y_full = sin.(x_full)
        y_view = @view y_full[:]
        prob_view = SampledIntegralProblem(y_view, x_full)
        sol_view = solve(prob_view, TrapezoidalRule())
        prob_regular = SampledIntegralProblem(y_full, x_full)
        sol_regular = solve(prob_regular, TrapezoidalRule())
        @test sol_view.u ≈ sol_regular.u
    end
end
