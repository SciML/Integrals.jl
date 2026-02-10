using Integrals
using Integrals: IntegralVerbosity
using FastGaussQuadrature
using StaticArrays
using Test

f = (x, p) -> prod(y -> cos(p * y), x)
exact_f = (lb, ub, p) -> prod(lu -> (sin(p * lu[2]) - sin(p * lu[1])) / p, zip(lb, ub))

# single dim

"""
    trapz(n::Integer)

Return the weights and nodes on the standard interval [-1,1] of the [trapezoidal
rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).
"""
function trapz(n::Integer)
    @assert n > 1
    r = range(-1, 1, length = n)
    x = collect(r)
    halfh = step(r) / 2
    h = step(r)
    w = [(i == 1) || (i == n) ? halfh : h for i in 1:n]
    return (x, w)
end

alg = QuadratureRule(trapz, n = 1000)

lb = -1.2
ub = 3.5
p = 2.0

prob = IntegralProblem(f, (lb, ub), p)
u = solve(prob, alg).u

@test u ≈ exact_f(lb, ub, p) rtol = 1.0e-3

# multi-dim

# here we just form a tensor product of 1d rules to make a 2d rule
function trapz2(n)
    x, w = trapz(n)
    return [SVector(y, z) for (y, z) in Iterators.product(x, x)], w .* w'
end

alg = QuadratureRule(trapz2, n = 100)

lb = SVector(-1.2, -1.0)
ub = SVector(3.5, 3.7)
p = 1.2

prob = IntegralProblem(f, (lb, ub), p)
u = solve(prob, alg).u

@test u ≈ exact_f(lb, ub, p) rtol = 1.0e-3

# 1d with inf limits

g = (x, p) -> p / (x^2 + p^2)

alg = QuadratureRule(gausslegendre, n = 1000)

lb = -Inf
ub = Inf
p = 1.0

prob = IntegralProblem(g, (lb, ub), p)

@test solve(prob, alg).u ≈ pi rtol = 1.0e-4

# 1d with nout

g2 = (x, p) -> [p[1] / (x^2 + p[1]^2), p[2] / (x^2 + p[2]^2)]

alg = QuadratureRule(gausslegendre, n = 1000)

lb = -Inf
ub = Inf
p = (1.0, 1.3)

prob = IntegralProblem(g2, (lb, ub), p)

@test solve(prob, alg).u ≈ [pi, pi] rtol = 1.0e-4

#= derivative tests

using Zygote

function testf(lb, ub, p, f = f)
    prob = IntegralProblem(f, (lb, ub), p)
    solve(prob, QuadratureRule(trapz, n=200))[1]
end

lb = -1.2
ub = 2.0
p = 3.1

dp = Zygote.gradient(p -> testf(lb, ub, p), p)

@test dp ≈ f(ub, p)-f(lb, p) rtol=1e-4
=#

# Verbosity tests

using SciMLLogging

@testset "Verbosity Tests" begin
    # Test function
    test_f = (x, p) -> sin(p * x)
    test_prob = IntegralProblem(test_f, (-1.0, 1.0), 2.0)
    test_alg = QuadratureRule(gausslegendre, n = 100)

    # Test 1: Default verbosity (Standard preset)
    @testset "Default Verbosity" begin
        cache = init(test_prob, test_alg)
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 2: Explicit Standard preset
    @testset "Standard Preset" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity())
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 3: None preset (silent mode)
    @testset "None Preset" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.None()))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 3b: Minimal preset
    @testset "Minimal Preset" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.Minimal()))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 3c: Detailed preset
    @testset "Detailed Preset" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.Detailed()))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 3d: All preset
    @testset "All Preset" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.All()))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 4: Boolean verbose parameter (true)
    @testset "Boolean Verbose - True" begin
        cache = init(test_prob, test_alg; verbose = true)
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 5: Boolean verbose parameter (false)
    @testset "Boolean Verbose - False" begin
        cache = init(test_prob, test_alg; verbose = false)
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 6: Individual toggles
    @testset "Individual Toggles" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(
            cache_init = Silent(),
            domain_transformation = InfoLevel(),
            algorithm_selection = DebugLevel(),
            iteration_progress = Silent(),
            convergence_result = InfoLevel(),
            batch_mode = Silent(),
            buffer_allocation = Silent(),
            deprecations = WarnLevel()
        ))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 7: Group settings - solver group
    @testset "Solver Group" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(solver = InfoLevel()))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 8: Group settings - setup group
    @testset "Setup Group" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(setup = InfoLevel()))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 9: Group settings - debug group
    @testset "Debug Group" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(debug = DebugLevel()))
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 10: Verbosity with solve() directly (not using cache)
    @testset "Direct Solve with Verbosity" begin
        sol = solve(test_prob, test_alg; verbose = IntegralVerbosity())
        @test sol.retcode == ReturnCode.Success

        sol = solve(test_prob, test_alg; verbose = false)
        @test sol.retcode == ReturnCode.Success

        sol = solve(test_prob, test_alg; verbose = true)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 11: Multiple verbosity levels
    @testset "Multiple Verbosity Levels" begin
        # Test Silent
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(
            cache_init = Silent()
        ))
        @test cache.verbosity isa IntegralVerbosity

        # Test InfoLevel
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(
            cache_init = InfoLevel()
        ))
        @test cache.verbosity isa IntegralVerbosity

        # Test DebugLevel
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(
            cache_init = DebugLevel()
        ))
        @test cache.verbosity isa IntegralVerbosity

        # Test WarnLevel
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(
            deprecations = WarnLevel()
        ))
        @test cache.verbosity isa IntegralVerbosity
    end

    # Test 12: Verbosity with multi-dimensional integrals
    @testset "Verbosity with Multi-dimensional" begin
        f_2d = (x, p) -> prod(y -> cos(p * y), x)
        lb_2d = SVector(-1.0, -1.0)
        ub_2d = SVector(1.0, 1.0)
        prob_2d = IntegralProblem(f_2d, (lb_2d, ub_2d), 1.0)

        # Use trapz2 from the earlier test
        function trapz2_test(n)
            r = range(-1, 1, length = n)
            x = collect(r)
            halfh = step(r) / 2
            h = step(r)
            w = [(i == 1) || (i == n) ? halfh : h for i in 1:n]
            return [SVector(y, z) for (y, z) in Iterators.product(x, x)], w .* w'
        end

        alg_2d = QuadratureRule(trapz2_test, n = 50)

        cache = init(prob_2d, alg_2d; verbose = IntegralVerbosity())
        @test cache.verbosity isa IntegralVerbosity
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
    end

    # Test 13: Verbosity persists in cache
    @testset "Verbosity Persistence in Cache" begin
        cache = init(test_prob, test_alg; verbose = IntegralVerbosity(
            convergence_result = InfoLevel()
        ))

        # Check that verbosity is stored
        @test cache.verbosity isa IntegralVerbosity

        # Solve and check verbosity still exists
        sol = solve!(cache)
        @test cache.verbosity isa IntegralVerbosity
        @test sol.retcode == ReturnCode.Success
    end

    # Test 14: All SciMLLogging presets
    @testset "All SciMLLogging Presets" begin
        # Test None
        sol = solve(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.None()))
        @test sol.retcode == ReturnCode.Success

        # Test Minimal
        sol = solve(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.Minimal()))
        @test sol.retcode == ReturnCode.Success

        # Test Standard
        sol = solve(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.Standard()))
        @test sol.retcode == ReturnCode.Success

        # Test Detailed
        sol = solve(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.Detailed()))
        @test sol.retcode == ReturnCode.Success

        # Test All
        sol = solve(test_prob, test_alg; verbose = IntegralVerbosity(SciMLLogging.All()))
        @test sol.retcode == ReturnCode.Success
    end
end
