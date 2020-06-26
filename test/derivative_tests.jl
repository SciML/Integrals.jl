using Quadrature, Zygote, FiniteDiff, ForwardDiff
using Test

### One Dimensional

f(x,p) = sum(sin.(p[1] * x))
lb = 1.0
ub = 3.0
p = 2.0
prob = QuadratureProblem(f,lb,ub,p)
sol = solve(prob,QuadGKJL(),reltol=1e-3,abstol=1e-3)

function testf(lb,ub,p)
    prob = QuadratureProblem(f,lb,ub,p)
    solve(prob,QuadGKJL(),reltol=1e-3,abstol=1e-3)[1]
end
dlb1,dub1,dp1 = Zygote.gradient(testf,lb,ub,p)
dlb2 = FiniteDiff.finite_difference_derivative(lb->testf(lb,ub,p),lb)
dub2 = FiniteDiff.finite_difference_derivative(ub->testf(lb,ub,p),ub)
dp2 = FiniteDiff.finite_difference_derivative(p->testf(lb,ub,p),p)

@test dlb1 ≈ dlb2
@test dub1 ≈ dub2
@test dp1 ≈ dp2

### N-dimensional

f(x,p) = sum(sin.(x .* p))
lb = ones(2)
ub = 3ones(2)
p = [1.5,2.0]
prob = QuadratureProblem(f,lb,ub,p)
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)

function testf(lb,ub,p)
    prob = QuadratureProblem(f,lb,ub,p)
    solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)[1]
end
dlb1,dub1,dp1 = Zygote.gradient(testf,lb,ub,p)
dlb2 = FiniteDiff.finite_difference_gradient(lb->testf(lb,ub,p),lb)
dub2 = FiniteDiff.finite_difference_gradient(ub->testf(lb,ub,p),ub)
dp2 = FiniteDiff.finite_difference_gradient(p->testf(lb,ub,p),p)

@test_broken dlb1 ≈ dlb2
@test_broken dub1 ≈ dub2
@test dp1 ≈ dp2

### N-dimensional N-out

f(x,p) = sin.(x .* p)
lb = ones(2)
ub = 3ones(2)
p = [1.5,2.0]
prob = QuadratureProblem(f,lb,ub,p)
sol = solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3)

function testf(lb,ub,p)
    prob = QuadratureProblem(f,lb,ub,p)
    sum(solve(prob,HCubatureJL(),reltol=1e-3,abstol=1e-3))
end
dlb1,dub1,dp1 = Zygote.gradient(testf,lb,ub,p)
dlb2 = FiniteDiff.finite_difference_gradient(lb->testf(lb,ub,p),lb)
dub2 = FiniteDiff.finite_difference_gradient(ub->testf(lb,ub,p),ub)
dp2 = FiniteDiff.finite_difference_gradient(p->testf(lb,ub,p),p)

@test_broken dlb1 ≈ dlb2
@test_broken dub1 ≈ dub2
@test dp1 ≈ dp2
