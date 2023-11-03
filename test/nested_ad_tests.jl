using Integrals, ForwardDiff, Cubature, Test

my_parameters = [1.0, 2.0]
my_function(x, p) = x^2 + p[1]^3 * x + p[2]^2
function my_integration(p)
    my_problem = IntegralProblem(my_function, -1.0, 1.0, p)
    # return solve(my_problem, HCubatureJL(), reltol=1e-3, abstol=1e-3)  # Works
    return solve(my_problem, CubatureJLh(), reltol=1e-3, abstol=1e-3)  # Errors
end
my_solution = my_integration(my_parameters)
@test ForwardDiff.jacobian(my_integration, my_parameters) == [0.0 8.0]
@test ForwardDiff.jacobian(x->ForwardDiff.jacobian(my_integration, x), my_parameters) == [0.0 0.0
                                                                                          0.0 4.0]
