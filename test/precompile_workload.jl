using Integrals
using Test

@testset "Precompile workload" begin
    prob = IntegralProblem((x, p) -> x^2, (0.0, 1.0))
    sol = solve(prob, QuadGKJL(); reltol = 1.0e-10, abstol = 1.0e-10)

    @test sol.u ≈ 1 / 3 rtol = 1.0e-10
end
