using Integrals, Test
@testset "Sampled Integration" begin
    lb = 0.4
    ub = 1.1
    npoints = 1000

    grid1 = range(lb, ub, length = npoints)
    grid2 = rand(npoints) .* (ub - lb) .+ lb
    grid2 = [lb; sort(grid2); ub]

    exact_sols = [1 / 6 * (ub^6 - lb^6), sin(ub) - sin(lb)]
    for method in [TrapezoidalRule] # Simpson's later
        for grid in [grid1, grid2]
            for (i, f) in enumerate([x -> x^5, x -> cos(x)])
                exact = exact_sols[i]
                # single dimensional y
                y = f.(grid)
                prob = SampledIntegralProblem(y, grid)
                error = solve(prob, method()).u .- exact
                @test all(error .< 10^-4)

                # along dim=2
                y = f.([grid grid]')
                prob = SampledIntegralProblem(y, grid; dim = 2)
                error = solve(prob, method()).u .- exact
                @test all(error .< 10^-4)
            end
        end
    end
end

@testset "Caching interface" begin
    x = 0.0:0.1:1.0
    y = sin.(x)

    prob = SampledIntegralProblem(y, x)
    alg = TrapezoidalRule()

    cache = init(prob, alg)
    sol1 = solve!(cache)

    @test sol1 == solve(prob, alg)

    cache.y = cos.(x)   # use .= to update in-place
    sol2 = solve!(cache)

    @test sol2 == solve(SampledIntegralProblem(cache.y, cache.x), alg)

    cache.x = 0.0:0.2:2.0
    cache.y = sin.(cache.x)
    sol3 = solve!(cache)

    @test sol3 == solve(SampledIntegralProblem(cache.y, cache.x), alg)

    x = 0.0:0.1:1.0
    y = sin.(x) .* cos.(x')

    prob = SampledIntegralProblem(y, x)
    alg = TrapezoidalRule()

    cache = init(prob, alg)
    sol1 = solve!(cache)

    @test sol1 == solve(prob, alg)

    cache.dim = 1
    sol2 = solve!(cache)

    @test sol2 == solve(SampledIntegralProblem(y, x, dim = 1), alg)
end
