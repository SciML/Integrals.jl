using Quadrature
using Cubature, Cuba
using Test

singledim_algs = [
        QuadGKJL(),
        HCubatureJL(),
        VEGAS(),
        CubatureJLh(),
        CubatureJLp(),
        CubaVegas(),
        CubaSUAVE()
        ]

multidim_algs = [
        HCubatureJL(),
        VEGAS(),
        CubatureJLh(),
        CubatureJLp(),
        CubaVegas(),
        CubaSUAVE(),
        CubaDivonne(),
        CubaCuhre()
        ]

integrands = [
              (x,p) -> 1,
              (x,p) -> sum(x),
              (x,p) -> sum(sin.(x))
             ]

prob = QuadratureProblem(integrands[1],ones(2),3ones(2))
_sol = solve(prob,HCubatureJL())
solve(prob,CubatureJLh(),reltol=1e-3,abstol=1e-3)

@testset "Standard Single Dimension Integrands" begin
    for f in integrands
        prob = QuadratureProblem(f,1,3)
        _sol = solve(prob,HCubatureJL())
        for alg in singledim_algs
            @info("Dimension = 1, Alg = $alg")
            sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
            @test sol.u ≈ _sol.u rtol = 1e-2
        end
    end
end

@testset "Standard Integrands" begin
    for f in integrands
        for dim = 1:5
            prob = QuadratureProblem(f,ones(dim),3ones(dim))
            _sol = solve(prob,CubatureJLh())
            for alg in multidim_algs
                if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre) ||
                    dim > 3 && alg isa VEGAS
                    continue
                end
                @info("Dimension = $dim, Alg = $alg")
                sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
                @test sol.u ≈ _sol.u rtol = 1e-2
            end
        end
    end
end

batch_f(f) = (pts) -> begin

    npts = size(pts, 1)
    ndims = size(pts, 2)
    fevals = zeros(npts)

    for i = 1:npts
        p = vec(pts[i,:])
        fevals[i] = f(p)
    end

    fevals
end

@testset "Batched Integrands" begin

    for f in integrands

        f2 = batch_f(f)

        for dim = 1:5
            @info("Dimension = $dim")
            v, _ = vegas(f2, zeros(dim), ones(dim), batch = true)
            h, _ = hcubature(f, zeros(dim), ones(dim))
            @test isapprox(v, h, rtol = 1e-2)
        end
    end
end
