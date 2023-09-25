using Integrals, Test
@testset "Sampled Integration" begin
    lb = 0.4
    ub = 1.1
    npoints = 1000

    grid1 = range(lb, ub, length = npoints)
    grid2 = rand(npoints) .* (ub - lb) .+ lb
    grid2 = [lb; sort(grid2); ub]

    exact_sols = [1 / 6 * (ub^6 - lb^6), sin(ub) - sin(lb)]
    exact_sols_cumulative = [[
        [1 / 6 * (x^6 - lb^6) for x in grid],
        [sin(x) - sin(lb) for x in grid],
    ] for grid in [grid1, grid2]]
    for method in [TrapezoidalRule] # Simpson's later
        for (j, grid) in enumerate([grid1, grid2])
            for (i, f) in enumerate([x -> x^5, x -> cos(x)])
                exact = exact_sols[i]
                exact_cum = exact_sols_cumulative[j]
                # single dimensional y
                y = f.(grid)
                prob = SampledIntegralProblem(y, grid)
                error = solve(prob, method()).u .- exact
                error_cum = solve(prob, method(); cumulative = true).u .- exact_cum[i]
                @test error < 10^-4
                @test all(error_cum .< 10^-2)

                # along dim=2
                y = f.([grid grid]')
                prob = SampledIntegralProblem(y, grid; dim = 2)
                error = solve(prob, method()).u .- exact
                error_cum = solve(prob, method(); cumulative = true).u .-
                            [exact_cum[i] exact_cum[i]]'
                @test all(error .< 10^-4)
                @test all(error_cum .< 10^-2)
            end
        end
    end
end

@testset "Caching interface" begin
    x = 0.0:0.1:1.0
    y = sin.(x)

    function test_interface(x, y, cumulative)
        prob = SampledIntegralProblem(y, x)
        alg = TrapezoidalRule()

        cache = init(prob, alg; cumulative)
        sol1 = solve!(cache)
        @test sol1 == solve(prob, alg; cumulative)

        cache.y = cos.(x)   # use .= to update in-place
        sol2 = solve!(cache)
        @test sol2 == solve(SampledIntegralProblem(cache.y, cache.x), alg; cumulative)

        cache.x = 0.0:0.2:2.0
        cache.y = sin.(cache.x)
        sol3 = solve!(cache)
        @test sol3 == solve(SampledIntegralProblem(cache.y, cache.x), alg; cumulative)

        x = 0.0:0.1:1.0
        y = sin.(x) .* cos.(x')

        prob = SampledIntegralProblem(y, x)
        alg = TrapezoidalRule()

        cache = init(prob, alg; cumulative)
        sol1 = solve!(cache)
        @test sol1 == solve(prob, alg; cumulative)

        cache.dim = 1
        sol2 = solve!(cache)
        @test sol2 == solve(SampledIntegralProblem(y, x, dim = 1), alg; cumulative)
    end
    @testset "Total Integral" begin
        test_interface(x, y, false)
    end
    @testset "Cumulative Integral" begin
        test_interface(x, y, true)
    end
end
