using Integrals
using JET
using Test

@testset "JET static analysis" begin
    @testset "QuadGKJL" begin
        f = (x, p) -> x^2
        prob = IntegralProblem(f, (0.0, 1.0))
        rep = @report_opt target_modules = (Integrals,) solve(prob, QuadGKJL())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "HCubatureJL" begin
        f = (x, p) -> x[1]^2 + x[2]^2
        prob = IntegralProblem(f, ([0.0, 0.0], [1.0, 1.0]))
        rep = @report_opt target_modules = (Integrals,) solve(prob, HCubatureJL())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "SampledIntegralProblem with TrapezoidalRule" begin
        x = range(0, 1, length = 100)
        y = x .^ 2
        prob = SampledIntegralProblem(y, x)
        rep = @report_opt target_modules = (Integrals,) solve(prob, TrapezoidalRule())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "SampledIntegralProblem with SimpsonsRule" begin
        x = range(0, 1, length = 101)
        y = x .^ 2
        prob = SampledIntegralProblem(y, x)
        rep = @report_opt target_modules = (Integrals,) solve(prob, SimpsonsRule())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "Infinite bounds transformation" begin
        f = (x, p) -> exp(-x^2)
        prob = IntegralProblem(f, (0.0, Inf))
        rep = @report_opt target_modules = (Integrals,) solve(prob, QuadGKJL())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "VEGAS" begin
        # VEGAS has some inherent type instability issues due to:
        # 1. Captured variables in closures (necessary for in-place operations)
        # 2. Runtime dispatch for integrand type checking
        # We verify the number of issues is bounded and doesn't regress
        f = (x, p) -> x^2
        prob = IntegralProblem(f, (0.0, 1.0))
        rep = @report_opt target_modules = (Integrals,) solve(prob, VEGAS())
        # Allow up to 2 reports (captured variable + runtime dispatch for type check)
        @test length(JET.get_reports(rep)) <= 2
    end
end
