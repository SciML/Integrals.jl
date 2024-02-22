using Integrals, FiniteDiff, ForwardDiff, Cubature, Cuba, Zygote, Test

my_parameters = [1.0, 2.0]
my_function(x, p) = x^2 + p[1]^3 * x + p[2]^2
function my_integration(p)
    my_problem = IntegralProblem(my_function, -1.0, 1.0, p)
    # return solve(my_problem, HCubatureJL(), reltol=1e-3, abstol=1e-3)  # Works
    return solve(my_problem, CubatureJLh(), reltol = 1e-3, abstol = 1e-3)  # Errors
end
my_solution = my_integration(my_parameters)
@test ForwardDiff.jacobian(my_integration, my_parameters) == [0.0 8.0]
@test ForwardDiff.jacobian(x -> ForwardDiff.jacobian(my_integration, x), my_parameters) ==
      [0.0 0.0
       0.0 4.0]

ff(x, p) = sum(sin.(x .* p))
lb = ones(2)
ub = 3ones(2)
p = [1.5, 2.0]

function testf(p)
    prob = IntegralProblem(ff, lb, ub, p)
    sin(solve(prob, CubaCuhre(), reltol = 1e-6, abstol = 1e-6)[1])
end

hp1 = FiniteDiff.finite_difference_hessian(testf, p)
hp2 = ForwardDiff.hessian(testf, p)
@test hp1≈hp2 atol=1e-4

ff2(x, p) = x * p[1] .+ p[2] * p[3]
lb = 1.0
ub = 3.0
p = [2.0, 3.0, 4.0]
_ff3 = BatchIntegralFunction(ff2)
prob = IntegralProblem(_ff3, lb, ub, p)

function testf3(lb, ub, p; f = _ff3)
    prob = IntegralProblem(_ff3, lb, ub, p)
    solve(prob, CubatureJLh(); reltol = 1e-3, abstol = 1e-3)[1]
end

dp1 = ForwardDiff.gradient(p -> testf3(lb, ub, p), p)
dp2 = Zygote.gradient(p -> testf3(lb, ub, p), p)[1]
dp3 = FiniteDiff.finite_difference_gradient(p -> testf3(lb, ub, p), p)

@test dp1 ≈ dp3
@test dp2 ≈ dp3
