using Integrals
using Cuba, Cubature, Arblib, FastGaussQuadrature, MCIntegration
using Test

max_dim_test = 2
max_nout_test = 2
reltol = 1e-3
abstol = 1e-3

alg_req = Dict(
    QuadGKJL() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = 1,
        allows_iip = true),
    QuadratureRule(gausslegendre, n = 50) => (
        nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false),
    GaussLegendre() => (nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false),
    HCubatureJL() => (nout = Inf, allows_batch = false, min_dim = 1,
        max_dim = Inf, allows_iip = true),
    # VEGAS() => (nout = 1, allows_batch = true, min_dim = 2, max_dim = Inf,
    #     allows_iip = true),
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
        nout = 1, allows_batch = false, min_dim = 1, max_dim = 1, allows_iip = true)
)

integrands = [
    (x, p) -> 1.0,
    (x, p) -> x isa Number ? cos(x) : prod(cos.(x))
]
iip_integrands = [(dx, x, p) -> (dx .= f(x, p)) for f in integrands]

integrands_v = [(x, p, nout) -> collect(1.0:nout)
                (x, p, nout) -> integrands[2](x, p) * collect(1.0:nout)]
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
            prob = IntegralProblem(integrands[i], lb, ub)
            @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
            sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                prob = IntegralProblem(integrands[i], lb, ub)
                if dim > req.max_dim || dim < req.min_dim
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                prob = IntegralProblem(iip_integrands[i], lb, ub)
                if dim > req.max_dim || dim < req.min_dim || !req.allows_iip
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
            prob = IntegralProblem(batch_f(integrands[i]), lb, ub, batch = 1000)
            if req.min_dim > 1 || !req.allows_batch
                continue
            end
            @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
            sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                prob = IntegralProblem(batch_f(integrands[i]), lb, ub, batch = 1000)
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                prob = IntegralProblem(batch_iip_f(integrands[i]), lb, ub, batch = 1000)
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch ||
                   !req.allows_iip
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                prob = IntegralProblem((x, p) -> integrands_v[i](x, p, nout), lb, ub,
                    nout = nout)
                if req.min_dim > 1 || req.nout < nout
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                    prob = IntegralProblem((x, p) -> integrands_v[i](x, p, nout), lb, ub,
                        nout = nout)
                    @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                    sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                    prob = IntegralProblem(
                        (dx, x, p) -> iip_integrands_v[i](dx, x, p, nout),
                        lb, ub, nout = nout)
                    if dim > req.max_dim || dim < req.min_dim || req.nout < nout ||
                       !req.allows_iip
                        continue
                    end
                    @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                    sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
            prob = IntegralProblem(batch_f_v(integrands_v[i], nout), lb, ub, batch = 1000,
                nout = nout)
            if req.min_dim > 1 || !req.allows_batch || req.nout < nout
                continue
            end
            @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
            sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                prob = IntegralProblem(batch_f_v(integrands_v[i], nout), lb, ub,
                    batch = 1000,
                    nout = nout)
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch ||
                   req.nout < nout
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
                prob = IntegralProblem(batch_iip_f_v(integrands_v[i], nout), lb, ub,
                    batch = 10, nout = nout)
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch ||
                   !req.allows_iip || req.nout < nout
                    continue
                end
                @info "Alg = $(nameof(typeof(alg))), Integrand = $i, Dimension = $dim, Output Dimension = $nout"
                sol = solve(prob, alg, reltol = reltol, abstol = abstol)
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
            prob = IntegralProblem(integrands[i], lb, ub, p)
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
