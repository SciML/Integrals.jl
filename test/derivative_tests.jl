using Integrals, Zygote, FiniteDiff, ForwardDiff#, SciMLSensitivity
using Cuba, Cubature
using Test

### One Dimensional

f(x, p) = sum(sin.(p[1] * x))
lb = 1.0
ub = 3.0
p = 2.0
prob = IntegralProblem(f, lb, ub, p)
sol = solve(prob, QuadGKJL(), reltol = 1e-3, abstol = 1e-3)

function testf(lb, ub, p)
    prob = IntegralProblem(f, lb, ub, p)
    sin(solve(prob, QuadGKJL(), reltol = 1e-3, abstol = 1e-3)[1])
end
dlb1, dub1, dp1 = Zygote.gradient(testf, lb, ub, p)
dlb2 = FiniteDiff.finite_difference_derivative(lb -> testf(lb, ub, p), lb)
dub2 = FiniteDiff.finite_difference_derivative(ub -> testf(lb, ub, p), ub)
dp2 = FiniteDiff.finite_difference_derivative(p -> testf(lb, ub, p), p)

# dlb3 = ForwardDiff.derivative(lb->testf(lb,ub,p),lb)
# dub3 = ForwardDiff.derivative(ub->testf(lb,ub,p),ub)
dp3 = ForwardDiff.derivative(p -> testf(lb, ub, p), p)

#@test dlb1 ≈ dlb3
#@test dub1 ≈ dub3
@test dp1 ≈ dp3

### N-dimensional

f(x, p) = sum(sin.(x .* p))
lb = ones(2)
ub = 3ones(2)
p = [1.5, 2.0]
prob = IntegralProblem(f, lb, ub, p)
sol = solve(prob, CubaCuhre(), reltol = 1e-3, abstol = 1e-3)

function testf(lb, ub, p)
    prob = IntegralProblem(f, lb, ub, p)
    sin(solve(prob, CubaCuhre(), reltol = 1e-6, abstol = 1e-6)[1])
end

function testf2(lb, ub, p)
    prob = IntegralProblem(f, lb, ub, p)
    sin(solve(prob, HCubatureJL(), reltol = 1e-6, abstol = 1e-6)[1])
end

dlb1, dub1, dp1 = Zygote.gradient(testf, lb, ub, p)
dlb2 = FiniteDiff.finite_difference_gradient(lb -> testf(lb, ub, p), lb)
dub2 = FiniteDiff.finite_difference_gradient(ub -> testf(lb, ub, p), ub)
dp2 = FiniteDiff.finite_difference_gradient(p -> testf(lb, ub, p), p)

@test_broken dlb1 ≈ dlb2
@test_broken dub1 ≈ dub2
@test dp1 ≈ dp2

# dlb3 = ForwardDiff.gradient(lb->testf(lb,ub,p),lb)
# dub3 = ForwardDiff.gradient(ub->testf(lb,ub,p),ub)
dp3 = ForwardDiff.gradient(p -> testf2(lb, ub, p), p)

#@test dlb1 ≈ dlb3
#@test dub1 ≈ dub3
@test dp1 ≈ dp3

# dlb4 = ForwardDiff.gradient(lb->testf(lb,ub,p),lb)
# dub4 = ForwardDiff.gradient(ub->testf(lb,ub,p),ub)
dp4 = ForwardDiff.gradient(p -> testf(lb, ub, p), p)

#@test dlb1 ≈ dlb4
#@test dub1 ≈ dub4
@test dp1 ≈ dp4

### N-dimensional N-out

f(x, p) = sin.(x .* p)
lb = ones(2)
ub = 3ones(2)
p = [1.5, 2.0]
prob = IntegralProblem(f, lb, ub, p, nout = 2)
sol = solve(prob, CubaCuhre(), reltol = 1e-3, abstol = 1e-3)

function testf(lb, ub, p)
    prob = IntegralProblem(f, lb, ub, p, nout = 2)
    sin(sum(solve(prob, CubaCuhre(), reltol = 1e-6, abstol = 1e-6)))
end

function testf2(lb, ub, p)
    prob = IntegralProblem(f, lb, ub, p, nout = 2)
    sin(sum(solve(prob, HCubatureJL(), reltol = 1e-6, abstol = 1e-6)))
end

dlb1, dub1, dp1 = Zygote.gradient(testf, lb, ub, p)
dlb2 = FiniteDiff.finite_difference_gradient(lb -> testf(lb, ub, p), lb)
dub2 = FiniteDiff.finite_difference_gradient(ub -> testf(lb, ub, p), ub)
dp2 = FiniteDiff.finite_difference_gradient(p -> testf(lb, ub, p), p)

@test_broken dlb1 ≈ dlb2
@test_broken dub1 ≈ dub2
@test dp1 ≈ dp2

# dlb3 = ForwardDiff.gradient(lb->testf(lb,ub,p),lb)
# dub3 = ForwardDiff.gradient(ub->testf(lb,ub,p),ub)
dp3 = ForwardDiff.gradient(p -> testf2(lb, ub, p), p)

#@test dlb1 ≈ dlb3
#@test dub1 ≈ dub3
@test dp1 ≈ dp3

# dlb4 = ForwardDiff.gradient(lb->testf(lb,ub,p),lb)
# dub4 = ForwardDiff.gradient(ub->testf(lb,ub,p),ub)
dp4 = ForwardDiff.gradient(p -> testf(lb, ub, p), p)

#@test dlb1 ≈ dlb4
#@test dub1 ≈ dub4
@test dp1 ≈ dp4

### Batch Single dim
f(x, p) = x * p[1] .+ p[2] * p[3]  # scalar integrand

lb = 1.0
ub = 3.0
p = [2.0, 3.0, 4.0]
prob = IntegralProblem(f, lb, ub, p)

function testf3(lb, ub, p; f = f)
    prob = IntegralProblem(f, lb, ub, p, batch = 10, nout = 1)
    solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)[1]
end

dp1 = ForwardDiff.gradient(p -> testf3(lb, ub, p), p)
dp2 = Zygote.gradient(p -> testf3(lb, ub, p), p)[1] # TODO fix: LoadError: DimensionMismatch("variable with size(x) == (1, 15) cannot have a gradient with size(dx) == (15,)")
dp3 = FiniteDiff.finite_difference_gradient(p -> testf3(lb, ub, p), p)

@test dp1 ≈ dp3 #passes
@test dp2 ≈ dp3 #passes

### Batch single dim, nout
f(x, p) = (x' * p[1] .+ p[2] * p[3]) .* [1; 2]

lb = 1.0
ub = 3.0
p = [2.0, 3.0, 4.0]
prob = IntegralProblem(f, lb, ub, p)

function testf3(lb, ub, p; f = f)
    prob = IntegralProblem(f, lb, ub, p, batch = 10, nout = 2)
    sum(solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3))
end

dp1 = ForwardDiff.gradient(p -> testf3(lb, ub, p), p)
dp2 = Zygote.gradient(p -> testf3(lb, ub, p), p)[1]
dp3 = FiniteDiff.finite_difference_gradient(p -> testf3(lb, ub, p), p)

@test dp1 ≈ dp3 #passes
@test dp2 ≈ dp3 #passes

### Batch multi dim
f(x, p) = x[1, :] * p[1] .+ p[2] * p[3]

lb = [1.0, 1.0]
ub = [3.0, 3.0]
p = [2.0, 3.0, 4.0]
prob = IntegralProblem(f, lb, ub, p)

function testf3(lb, ub, p; f = f)
    prob = IntegralProblem(f, lb, ub, p, batch = 10, nout = 1)
    solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)[1]
end

dp1 = ForwardDiff.gradient(p -> testf3(lb, ub, p), p)
dp2 = Zygote.gradient(p -> testf3(lb, ub, p), p)[1]
dp3 = FiniteDiff.finite_difference_gradient(p -> testf3(lb, ub, p), p)

@test dp1 ≈ dp3
@test dp2 ≈ dp3

### Batch multi dim nout
f(x, p) = x * p[1] .+ p[2] * p[3]

lb = [1.0, 1.0]
ub = [3.0, 3.0]
p = [2.0, 3.0, 4.0]
prob = IntegralProblem(f, lb, ub, p)

function testf3(lb, ub, p; f = f)
    prob = IntegralProblem(f, lb, ub, p, batch = 10, nout = 2)
    sum(solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3))
end

dp1 = ForwardDiff.gradient(p -> testf3(lb, ub, p), p)
dp2 = Zygote.gradient(p -> testf3(lb, ub, p), p)[1]
dp3 = FiniteDiff.finite_difference_gradient(p -> testf3(lb, ub, p), p)

@test dp1 ≈ dp3
@test dp2 ≈ dp3

## iip Batch multi dim
function g(dx, x, p)
    dx .= sum(x * p[1] .+ p[2] * p[3], dims = 1)
end

lb = [1.0, 1.0]
ub = [3.0, 3.0]
p = [2.0, 3.0, 4.0]
prob = IntegralProblem(g, lb, ub, p)

function testf3(lb, ub, p; f = g)
    prob = IntegralProblem(f, lb, ub, p, batch = 10, nout = 1)
    solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)[1]
end

testf3(lb, ub, p)

dp1 = ForwardDiff.gradient(p -> testf3(lb, ub, p), p)
dp2 = Zygote.gradient(p -> testf3(lb, ub, p), p)[1]
dp3 = FiniteDiff.finite_difference_gradient(p -> testf3(lb, ub, p), p)

@test dp1 ≈ dp3
@test dp2 ≈ dp3

## iip Batch mulit dim nout
function g(dx, x, p)
    dx .= x * p[1] .+ p[2] * p[3]
end

lb = [1.0, 1.0]
ub = [3.0, 3.0]
p = [2.0, 3.0, 4.0]
prob = IntegralProblem(g, lb, ub, p)

function testf3(lb, ub, p; f = g)
    prob = IntegralProblem(f, lb, ub, p, batch = 10, nout = 2)
    sum(solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3))
end

dp1 = ForwardDiff.gradient(p -> testf3(lb, ub, p), p)
dp2 = Zygote.gradient(p -> testf3(lb, ub, p), p)[1]
dp3 = FiniteDiff.finite_difference_gradient(p -> testf3(lb, ub, p), p)

@test dp1 ≈ dp3
@test dp2 ≈ dp3
