using Integrals, Test, FastGaussQuadrature

#=
f = (x, p) -> x^3 * sin(5x)
n = 250
nodes, weights = gausslegendre(n)
I = gauss_legendre(f, nothing, -1, 1, nodes, weights)
@test I ≈ 2 / (625) * (69sin(5) - 95cos(5))
I = Integrals.composite_gauss_legendre(f, nothing, -1, 1, nodes, weights, 2)
@test I ≈ 2 / (625) * (69sin(5) - 95cos(5))

f = (x, p) -> (x + p) * abs(x)
n = 100
nodes, weights = gausslegendre(n)
I = Integrals.gauss_legendre(f, 0.0, -2, 2, nodes, weights)
Ic = Integrals.composite_gauss_legendre(f, 6, -2, 2, nodes, weights, 5)
@inferred Integrals.gauss_legendre(f, 0.0, -2, 2, nodes, weights)
@inferred Integrals.composite_gauss_legendre(f, 6, -2, 2, nodes, weights, 5)
@test I≈0.0 atol=1e-6
@test Ic≈24 rtol=1e-4
=#

alg = GaussLegendre()
n = 250
nd, wt = gausslegendre(n)
@test alg.nodes == nd
@test alg.weights == wt
@test alg.subintervals == 1
alg = GaussLegendre(n = 125, subintervals = 3)
n = 125
nd, wt = gausslegendre(n)
@test alg.nodes == nd
@test alg.weights == wt
@test alg.subintervals == 3
@test typeof(alg).parameters[1]
nd, wt = gausslegendre(275)
alg = GaussLegendre(nodes = nd, weights = wt)
@test !typeof(alg).parameters[1]
@test alg.nodes == nd
@test alg.weights == wt
@test alg.subintervals == 1
alg = GaussLegendre(nodes = nd, weights = wt, subintervals = 20)
@test typeof(alg).parameters[1]
@test alg.nodes == nd
@test alg.weights == wt
@test alg.subintervals == 20

f = (x, p) -> 5x + sin(x) - p * exp(x)
prob = IntegralProblem(f, -5, 3, 3.3)
alg = GaussLegendre()
sol = solve(prob, alg)
@test isnothing(sol.chi)
# These tests don't pass with ChangeOfVariables
# @test sol.alg === alg
# @test sol.prob === prob
@test isnothing(sol.resid)
@test SciMLBase.successful_retcode(sol)
@test sol.u ≈ -exp(3) * 3.3 + 3.3 / exp(5) - 40 + cos(5) - cos(3)
alg = GaussLegendre(subintervals = 7)
sol = solve(prob, alg)
@test sol.u ≈ -exp(3) * 3.3 + 3.3 / exp(5) - 40 + cos(5) - cos(3)

f = (x, p) -> exp(-x^2)
prob = IntegralProblem(f, 0.0, Inf)
alg = GaussLegendre()
sol = solve(prob, alg)
@test sol.u ≈ sqrt(π) / 2
alg = GaussLegendre(subintervals = 1)
@test sol.u ≈ sqrt(π) / 2
alg = GaussLegendre(subintervals = 17)
@test sol.u ≈ sqrt(π) / 2

prob = IntegralProblem(f, -Inf, Inf)
alg = GaussLegendre()
sol = solve(prob, alg)
@test sol.u ≈ sqrt(π)
alg = GaussLegendre(subintervals = 1)
@test sol.u ≈ sqrt(π)
alg = GaussLegendre(subintervals = 17)
@test sol.u ≈ sqrt(π)

prob = IntegralProblem(f, -Inf, 0.0)
alg = GaussLegendre()
sol = solve(prob, alg)
@test sol.u ≈ sqrt(π) / 2
alg = GaussLegendre(subintervals = 1)
@test sol.u ≈ sqrt(π) / 2
alg = GaussLegendre(subintervals = 17)
@test sol.u ≈ sqrt(π) / 2

# Make sure broadcasting correctly handles the argument p
f = (x, p) -> 1 + x + x^p[1] - cos(x * p[2]) + exp(x) * p[3]
p = [0.3, 1.3, -0.5]
prob = IntegralProblem(f, 2.0, 6.3, p)
alg = GaussLegendre()
sol = solve(prob, alg)
@test sol.u ≈ -240.25235266303063249920743158729
alg = GaussLegendre(n = 500, subintervals = 17)
sol = solve(prob, alg)
@test sol.u ≈ -240.25235266303063249920743158729
