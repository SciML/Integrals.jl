# Differentiating Integrals

Integrals.jl is a fully differentiable quadrature library. Thus, it adds the
ability to perform automatic differentiation over any of the libraries that it
calls. It integrates with ForwardDiff.jl for forward-mode automatic differentiation
and Zygote.jl for reverse-mode automatic differentiation. For example:

``` @example AD
using Integrals, ForwardDiff, FiniteDiff, Zygote, IntegralsCuba
f(x,p) = sum(sin.(x .* p))
lb = ones(2)
ub = 3ones(2)
p = ones(2)

function testf(p)
    prob = IntegralProblem(f,lb,ub,p)
    sin(solve(prob,CubaCuhre(),reltol=1e-6,abstol=1e-6)[1])
end
testf(p)
#dp1 = Zygote.gradient(testf,p) #broken
dp2 = FiniteDiff.finite_difference_gradient(testf,p)
dp3 = ForwardDiff.gradient(testf,p)
dp2 ≈ dp3 #dp1[1] ≈ dp2 ≈ dp3
```