using Test
using ForwardDiff
using Integrals

a, b = 0.0, π

result = solve(IntegralProblem((x, p) -> sin(x), (0, π)), GaussTuran(5, 2, ForwardDiff))

expected_result = 2.0

@test isapprox(result, expected_result, atol=1e-6)

println("Test passed, result of integration: ", result)
