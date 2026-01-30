using Integrals, HAdaptiveIntegration
using Test

@testset "HAdaptiveIntegrationJL" begin
    reltol = 1.0e-6
    abstol = 1.0e-6

    @testset "1D orthotope" begin
        prob = IntegralProblem((x, p) -> sin(x), (0.0, 1.0))
        sol = solve(prob, HAdaptiveIntegrationJL(), reltol = reltol, abstol = abstol)
        @test sol.u ≈ 1 - cos(1.0) rtol = 1.0e-6
        @test sol.retcode == ReturnCode.Success
    end

    @testset "2D orthotope" begin
        prob = IntegralProblem((x, p) -> x[1] + x[2], (zeros(2), ones(2)))
        sol = solve(prob, HAdaptiveIntegrationJL(), reltol = reltol, abstol = abstol)
        @test sol.u ≈ 1.0 rtol = 1.0e-6
    end

    @testset "3D orthotope" begin
        prob = IntegralProblem((x, p) -> 1.0, (zeros(3), ones(3)))
        sol = solve(prob, HAdaptiveIntegrationJL(), reltol = reltol, abstol = abstol)
        @test sol.u ≈ 1.0 rtol = 1.0e-6
    end

    @testset "Triangle" begin
        prob = IntegralProblem(
            (x, p) -> 1.0, Triangle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
        sol = solve(prob, HAdaptiveIntegrationJL(), reltol = reltol, abstol = abstol)
        @test sol.u ≈ 0.5 rtol = 1.0e-6
    end

    @testset "Tetrahedron" begin
        prob = IntegralProblem(
            (x, p) -> 1.0,
            Tetrahedron(
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        sol = solve(prob, HAdaptiveIntegrationJL(), reltol = reltol, abstol = abstol)
        @test sol.u ≈ 1 / 6 rtol = 1.0e-6
    end

    @testset "With parameters" begin
        prob = IntegralProblem((x, p) -> p * x[1], (zeros(2), ones(2)), 3.0)
        sol = solve(prob, HAdaptiveIntegrationJL(), reltol = reltol, abstol = abstol)
        @test sol.u ≈ 1.5 rtol = 1.0e-6
    end

    @testset "Error estimate" begin
        prob = IntegralProblem((x, p) -> sin(x), (0.0, 1.0))
        sol = solve(prob, HAdaptiveIntegrationJL())
        @test sol.resid isa Number
        @test sol.resid >= 0
    end
end
