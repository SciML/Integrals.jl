using Integrals, Distributions, Test, StaticArrays

μ = [0.00, 0.00]
Σ = [0.4 0.0; 0.00 0.4]
d = MvNormal(μ, Σ)
m(x, p) = pdf(d, x)
prob = IntegralProblem(m, [-Inf, -Inf], [Inf, Inf])
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (1.00 - sol.u)^2 < 1e-6
@test_nowarn @inferred Integrals.transformation_if_inf(prob, Val(true))

μ = [0.00, 0.00]
Σ = [0.4 0.0; 0.00 0.4]
d = MvNormal(μ, Σ)
m(x, p) = pdf(d, x)
prob = IntegralProblem(m, [-Inf, 0.00], [Inf, Inf])
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (0.500 - sol.u)^2 < 1e-6
@test_nowarn @inferred Integrals.transformation_if_inf(prob, Val(true))

μ = [0.00, 0.00]
Σ = [0.4 0.0; 0.00 0.4]
d = MvNormal(μ, Σ)
m_iip(dx, x, p) = dx .= pdf(d, x)
prob = IntegralProblem(m_iip, [-Inf, 0.00], [Inf, Inf])
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (0.500 - sol.u[1])^2 < 1e-6

f(x, p) = pdf(Normal(0.00, 1.00), x)
prob = IntegralProblem(f, -Inf, Inf)
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (1.00 - sol.u)^2 < 1e-6
@test_nowarn @inferred Integrals.transformation_if_inf(prob, Val(true))

f(x, p) = pdf(Normal(0.00, 1.00), x)
prob = IntegralProblem(f, -Inf, 0.00)
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (0.50 - sol.u)^2 < 1e-6
@test_nowarn @inferred Integrals.transformation_if_inf(prob, Val(true))

f(x, p) = (1 / (x^2 + 1))
prob = IntegralProblem(f, 0.0, Inf)
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (pi / 2 - sol.u)^2 < 1e-6
@test_nowarn @inferred Integrals.transformation_if_inf(prob, Val(true))

# Type stability
μ = [0.00, 0.00]
Σ = [0.4 0.0; 0.00 0.4]
d = MvNormal(μ, Σ)
m2 = let d = d
    (x, p) -> pdf(d, x)
end

prob = IntegralProblem(m2, SVector(-Inf, -Inf), SVector(Inf, Inf))
@test_nowarn @inferred solve(prob, HCubatureJL(); do_inf_transformation = Val(true))

prob = @test_nowarn @inferred Integrals.transformation_if_inf(prob, Val(true))
@test_nowarn @inferred Integrals.__solvebp_call(prob, HCubatureJL(),
    Integrals.ReCallVJP(Integrals.ZygoteVJP()),
    prob.lb, prob.ub, prob.p)
