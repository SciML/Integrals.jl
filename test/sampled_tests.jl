using Integrals, Test
@testset "Sampled Integration" begin
    lb = 0.0
    ub = 1.0
    npoints = 1000
    for method = [Trapezoidal] # Simpson's later
        nouts = [1,2,1,2]
        for (i,f) = enumerate([(x,p) -> x^5, (x,p) -> [x^5, x^5], (out, x,p) -> (out[1] = x^5; out),  (out, x, p) -> (out[1] = x^5; out[2] = x^5; out)])

            exact = 1/6
            prob = IntegralProblem(f, lb, ub, nout=nouts[i])

            # AbstractRange
            error1 = solve(prob, method(npoints)).u .- exact
            @test all(error1 .< 10^-4)

            # AbstractVector equidistant
            error2 = solve(prob, method(collect(range(lb, ub, length=npoints)))).u .- exact
            @test all(error2 .< 10^-4)

            # AbstractVector irregular
            grid = rand(npoints)
            grid = [lb; sort(grid); ub]
            error3 = solve(prob, method(grid)).u .- exact
            @test all(error3 .< 10^-4)


        end
        exact = 1/6

        grid = rand(npoints)
        grid = [lb; sort(grid); ub]
        # single dimensional y
        y = grid .^ 5
        prob = IntegralProblem(y, lb, ub)
        error4 = solve(prob, method(grid, dim=1)).u .- exact
        @test all(error4 .< 10^-4) 

        # along dim=2
        y = ([grid grid]') .^ 5
        prob = IntegralProblem(y, lb, ub)
        error5 = solve(prob, method(grid, dim=2)).u .- exact 
        @test all(error5 .< 10^-4) 
    end
end