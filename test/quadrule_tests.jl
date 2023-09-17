using Integrals
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

prob = IntegralProblem(f, lb, ub, p)
u = solve(prob, alg).u

@test u≈exact_f(lb, ub, p) rtol=1e-3

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

prob = IntegralProblem(f, lb, ub, p)
u = solve(prob, alg).u

@test u≈exact_f(lb, ub, p) rtol=1e-3

# 1d with inf limits

g = (x, p) -> p / (x^2 + p^2)

alg = QuadratureRule(gausslegendre, n = 1000)

lb = -Inf
ub = Inf
p = 1.0

prob = IntegralProblem(g, lb, ub, p)

@test solve(prob, alg).u≈pi rtol=1e-4

# 1d with nout

g2 = (x, p) -> [p[1] / (x^2 + p[1]^2), p[2] / (x^2 + p[2]^2)]

alg = QuadratureRule(gausslegendre, n = 1000)

lb = -Inf
ub = Inf
p = (1.0, 1.3)

prob = IntegralProblem(g2, lb, ub, p)

@test solve(prob, alg).u≈[pi,pi] rtol=1e-4


#= derivative tests

using Zygote

function testf(lb, ub, p, f = f)
    prob = IntegralProblem(f, lb, ub, p)
    solve(prob, QuadratureRule(trapz, n=200))[1]
end

lb = -1.2
ub = 2.0
p = 3.1

dp = Zygote.gradient(p -> testf(lb, ub, p), p)

@test dp ≈ f(ub, p)-f(lb, p) rtol=1e-4
=#
