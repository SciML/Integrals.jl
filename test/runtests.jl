using Quadrature
using Cubature, Cuba
using Test

singledim_algs = [
        QuadGKJL(),
        HCubatureJL(),
        # VEGAS(),
        CubatureJLh(),
        CubatureJLp(),
        # CubaVegas(),
        CubaSUAVE()
        ]

multidim_algs = [
        HCubatureJL(),
        # VEGAS(),
        CubatureJLh(),
        CubatureJLp(),
        # CubaVegas(),
        CubaSUAVE(),
        CubaDivonne(),
        CubaCuhre()
        ]

singledim_batch_algs = [
        # VEGAS(),
        CubatureJLh(),
        CubatureJLp(),
        CubaVegas(),
        CubaSUAVE()
        ]

multidim_batch_algs = [
        # VEGAS(),
        CubatureJLh(),
        CubatureJLp(),
        # CubaVegas(),
        CubaSUAVE(),
        CubaDivonne(),
        CubaCuhre()
        ]

integrands = [
              (x,p) -> 1,
              (x,p) -> x isa Number ? cos(x) : prod(cos.(x))
             ]

iip_integrands = [ (dx,x,p)-> (dx .= f(x,p)) for f ∈ integrands]


exact_sol = [
                (ndim, nout, lb, ub) -> prod(ub-lb),
                (ndim, nout, lb, ub) -> prod(sin.(ub)-sin.(lb))
            ]

batch_f(f) = (pts,p) -> begin
  fevals = zeros(size(pts, 2))
  for i = 1:size(pts, 2)
     x = pts[:,i]
     fevals[i] = f(x,p)
  end
  fevals
end

batch_iip_f(f) = (fevals,pts,p) -> begin
  for i = 1:size(pts, 2)
     x = pts[:,i]
     fevals[i] = f(x,p)
  end
  nothing
end

prob = QuadratureProblem(integrands[1],ones(2),3ones(2))
_sol = solve(prob,HCubatureJL())
solve(prob,CubatureJLh(),reltol=1e-3,abstol=1e-3)

@testset "Standard Single Dimension Integrands" begin
    lb,ub = (1,3)
    for i in 1:length(integrands)
        prob = QuadratureProblem(integrands[i],lb,ub)
        # _sol = solve(prob,HCubatureJL())
        for alg in singledim_algs
            @info "Dimension = 1, Alg = $alg, Integrand = $i"
            sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
            @test sol.u ≈ exact_sol[i](1,1,lb,ub) rtol = 1e-2
        end
    end
end

@testset "Standard Integrands" begin
    for i in 1:length(integrands)
        for dim = 1:5
            lb, ub = (ones(dim), 3ones(dim))
            prob = QuadratureProblem(integrands[i],lb,ub)
            # _sol = solve(prob,CubatureJLh())
            for alg in multidim_algs
                if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre) ||
                    dim > 3 && alg isa VEGAS
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
                @test sol.u ≈ exact_sol[i](dim,1,lb,ub) rtol = 1e-2
            end
        end
    end
end

@testset "In-place Standard Integrands" begin
    for i in 1:length(iip_integrands)
        for dim = 1:5
            lb, ub = (ones(dim), 3ones(dim))
            prob = QuadratureProblem(iip_integrands[i],lb,ub)
            _sol = solve(prob,CubatureJLh())
            for alg in multidim_algs
                if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre) ||
                    dim > 3 && alg isa VEGAS # Large VEGAS omitted because it's slow!
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
                @test sol.u ≈ exact_sol[i](dim,1,lb,ub) rtol = 1e-2
            end
        end
    end
end
#
# @testset "Batched Single Dimension Integrands" begin
#     for i in 1:length(integrands)
#         _prob = QuadratureProblem(integrands[i],1,3)
#         _sol = solve(_prob,CubatureJLh())
#         prob = QuadratureProblem(batch_f(integrands[i]),1,3,batch=10)
#         for alg in singledim_batch_algs
#             @info "Dimension = 1, Alg = $alg, Integrand = $i"
#             sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
#             if !isapprox(_sol.u,sol.u,rtol = 1e-2)
#                 @info "$(alg) failed"
#                 @show sol.u,_sol.u
#                 @test_broken sol.u ≈ _sol.u rtol = 1e-2
#             else
#                 @test sol.u ≈ _sol.u rtol = 1e-2
#             end
#         end
#     end
# end
#
# @testset "Batched Standard Integrands" begin
#     for i in 1:length(integrands)
#         for dim = 1:5
#             _prob = QuadratureProblem(integrands[i],ones(dim),3ones(dim))
#             _sol = solve(_prob,CubatureJLh())
#             prob = QuadratureProblem(batch_f(integrands[i]),ones(dim),3ones(dim),batch=10)
#             for alg in multidim_batch_algs
#                 if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre) ||
#                    dim > 3 && alg isa VEGAS # Large VEGAS omitted because it's slow!
#                     continue
#                 end
#                 @info "Dimension = $dim, Alg = $alg, Integrand = $i"
#                 sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
#                 @test sol.u ≈ _sol.u rtol = 1e-2
#             end
#         end
#     end
# end
#
# @testset "In-Place Batched Standard Integrands" begin
#     for i in 1:length(iip_integrands)
#         for dim = 1:5
#             _prob = QuadratureProblem(iip_integrands[i],ones(dim),3ones(dim))
#             _sol = solve(_prob,CubatureJLh())
#             prob = QuadratureProblem(batch_iip_f(integrands[i]),ones(dim),3ones(dim),batch=10)
#             for alg in multidim_batch_algs
#                 if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre)
#                     continue
#                 end
#                 @info "Dimension = $dim, Alg = $alg, Integrand = $i"
#                 if alg isa VEGAS
#                     @test_broken sol = solve(prob,alg,reltol=1e-3,abstol=1e-3).retcode == :Success
#                 else
#                     sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
#                     @test sol.u ≈ _sol.u rtol = 1e-2
#                 end
#             end
#         end
#     end
# end
