using Integrals, Distributions, Test

μ = [0.00, 0.00]
Σ = [0.4 0.0; 0.00 0.4]
d = MvNormal(μ, Σ)
m(x, p) = pdf(d, x)
prob = IntegralProblem(m, [-Inf, -Inf], [Inf, Inf])
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (1.00 - sol.u)^2 < 1e-6

μ = [0.00, 0.00]
Σ = [0.4 0.0; 0.00 0.4]
d = MvNormal(μ, Σ)
m(x, p) = pdf(d, x)
prob = IntegralProblem(m, [-Inf, 0.00], [Inf, Inf])
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (0.500 - sol.u)^2 < 1e-6

f(x, p) = pdf(Normal(0.00, 1.00), x)
prob = IntegralProblem(f, -Inf, Inf)
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (1.00 - sol.u)^2 < 1e-6

f(x, p) = pdf(Normal(0.00, 1.00), x)
prob = IntegralProblem(f, -Inf, 0.00)
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (0.50 - sol.u)^2 < 1e-6

f(x, p) = (1 / (x^2 + 1))
prob = IntegralProblem(f, 0, Inf)
sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
@test (pi / 2 - sol.u)^2 < 1e-6
