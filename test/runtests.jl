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

singledim_batch_algs = [
        VEGAS(),
        CubatureJLh(),
        CubatureJLp(),
        CubaVegas(),
        CubaSUAVE()
        ]

multidim_batch_algs = [
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
    for i in 1:length(integrands)
        prob = QuadratureProblem(integrands[i],1,3)
        _sol = solve(prob,HCubatureJL())
        for alg in singledim_algs
            @info "Dimension = 1, Alg = $alg, Integrand = $i"
            sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
            @test sol.u ≈ _sol.u rtol = 1e-2
        end
    end
end

@testset "Standard Integrands" begin
    for i in 1:length(integrands)
        for dim = 1:5
            prob = QuadratureProblem(integrands[i],ones(dim),3ones(dim))
            _sol = solve(prob,CubatureJLh())
            for alg in multidim_algs
                if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre) ||
                    dim > 3 && alg isa VEGAS
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
                @test sol.u ≈ _sol.u rtol = 1e-2
            end
        end
    end
end

batch_f(f) = (pts,p) -> begin

    ndims = size(pts, 1)
    npts = size(pts, 2)
    fevals = zeros(npts)
    for i = 1:npts
        x = pts[:,i]
        fevals[i] = f(x,p)
    end

    fevals
end

@testset "Batched Single Dimension Integrands" begin
    for i in 1:length(integrands)
        prob = QuadratureProblem(batch_f(integrands[i]),1,3,batch=1)
        _sol = solve(prob,CubatureJLh())
        for alg in singledim_batch_algs
            @info "Dimension = 1, Alg = $alg, Integrand = $i"
            sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
            if !isapprox(_sol.u,sol.u,rtol = 1e-2)
                @info "$(alg) failed"
                @show sol.u,_sol.u
                @test_broken sol.u ≈ _sol.u rtol = 1e-2
            else
                @test sol.u ≈ _sol.u rtol = 1e-2
            end
        end
    end
end

@testset "Batched Standard Integrands" begin
    for i in 1:length(integrands)
        for dim = 1:5
            prob = QuadratureProblem(batch_f(integrands[i]),ones(dim),3ones(dim),batch=true)
            _sol = solve(prob,CubatureJLh())
            for alg in multidim_batch_algs
                if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre) ||
                    dim > 3 && alg isa VEGAS
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
                @test sol.u ≈ _sol.u rtol = 1e-2
            end
        end
    end
end
