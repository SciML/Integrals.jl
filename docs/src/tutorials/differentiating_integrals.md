# Differentiating Integrals

Integrals.jl is a fully differentiable quadrature library. Thus, it adds the
ability to perform automatic differentiation over any of the libraries that it
calls. It integrates with ForwardDiff.jl for forward-mode automatic differentiation
and Zygote.jl for reverse-mode automatic differentiation. For example:

```@example AD
using Integrals, ForwardDiff, FiniteDiff, Zygote
f(x, p) = sum(sin.(x .* p))
domain = (ones(2), 3ones(2)) # (lb, ub)
p = ones(2)

function testf(p)
    prob = IntegralProblem(f, domain, p)
    sin(solve(prob, HCubatureJL(), reltol = 1e-6, abstol = 1e-6)[1])
end
testf(p)
dp1 = Zygote.gradient(testf, p)
dp2 = FiniteDiff.finite_difference_gradient(testf, p)
dp3 = ForwardDiff.gradient(testf, p)
dp1[1] ≈ dp2 ≈ dp3
```
