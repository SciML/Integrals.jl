##
using Quadrature
using Cubature, Cuba
using Test

algs = [QuadGKJL(), HCubatureJL(), CubatureJLh(), CubatureJLp(), #VEGAS(), CubaVegas(),
        CubaSUAVE(),CubaDivonne(), CubaCuhre()]

alg_req=Dict(QuadGKJL()=>     (nout=1,   allows_batch=false, min_dim=1, max_dim=1,   allows_iip = false),
             HCubatureJL()=>  (nout=Inf, allows_batch=false, min_dim=1, max_dim=Inf, allows_iip = true ),
             VEGAS()=>        (nout=1,   allows_batch=true,  min_dim=1, max_dim=Inf, allows_iip = true ),
             CubatureJLh()=>  (nout=Inf, allows_batch=true,  min_dim=1, max_dim=Inf, allows_iip = true ),
             CubatureJLp()=>  (nout=Inf, allows_batch=true,  min_dim=1, max_dim=Inf, allows_iip = true ),
             CubaVegas()=>    (nout=Inf, allows_batch=true,  min_dim=1, max_dim=Inf, allows_iip = true ),
             CubaSUAVE()=>    (nout=Inf, allows_batch=true,  min_dim=1, max_dim=Inf, allows_iip = true ),
             CubaDivonne()=>  (nout=Inf, allows_batch=true,  min_dim=2, max_dim=Inf, allows_iip = true ),
             CubaCuhre()=>    (nout=Inf, allows_batch=true,  min_dim=2, max_dim=Inf, allows_iip = true ))

integrands = [
              (x,p) -> 1.0,
              (x,p) -> x isa Number ? cos(x) : prod(cos.(x))
             ]

iip_integrands = [ (dx,x,p)-> (dx .= f(x,p)) for f ∈ integrands]


exact_sol = [
                (ndim, nout, lb, ub) -> prod(ub-lb),
                (ndim, nout, lb, ub) -> prod(sin.(ub)-sin.(lb))
            ]

batch_f(f) = (pts,p) -> begin
  fevals = zero(pts)
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
        for alg in algs#singledim_algs
            if alg_req[alg].min_dim > 1
                continue
            end
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
            for alg in algs
                req = alg_req[alg]
                if dim > req.max_dim || dim < req.min_dim || alg isa QuadGKJL  #QuadGKJL requires numbers, not single element arrays
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
            for alg in algs
                req = alg_req[alg]
                if dim > req.max_dim || dim < req.min_dim || alg isa QuadGKJL  #QuadGKJL requires numbers, not single element arrays
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
                @test sol.u ≈ exact_sol[i](dim,1,lb,ub) rtol = 1e-2
            end
        end
    end
end

########## put vector valued function tests here ##############


@testset "Batched Single Dimension Integrands" begin
    (lb,ub) = (1,3)
    (ndim, nout) = (1,1)
    for i in 1:length(integrands)
        prob = QuadratureProblem(batch_f(integrands[i]),lb,ub,batch=10)
        for alg in algs
            req = alg_req[alg]
            if req.min_dim > 1 || !req.allows_batch
                continue
            end

            @info "Dimension = 1, Alg = $alg, Integrand = $i"
            sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
            @test sol.u ≈ exact_sol[i](ndim,nout,lb,ub) rtol = 1e-2
        end
    end
end
# #
# # @testset "Batched Standard Integrands" begin
# #     for i in 1:length(integrands)
# #         for dim = 1:5
# #             _prob = QuadratureProblem(integrands[i],ones(dim),3ones(dim))
# #             _sol = solve(_prob,CubatureJLh())
# #             prob = QuadratureProblem(batch_f(integrands[i]),ones(dim),3ones(dim),batch=10)
# #             for alg in multidim_batch_algs
# #                 if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre) ||
# #                    dim > 3 && alg isa VEGAS # Large VEGAS omitted because it's slow!
# #                     continue
# #                 end
# #                 @info "Dimension = $dim, Alg = $alg, Integrand = $i"
# #                 sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
# #                 @test sol.u ≈ _sol.u rtol = 1e-2
# #             end
# #         end
# #     end
# # end
# #
# # @testset "In-Place Batched Standard Integrands" begin
# #     for i in 1:length(iip_integrands)
# #         for dim = 1:5
# #             _prob = QuadratureProblem(iip_integrands[i],ones(dim),3ones(dim))
# #             _sol = solve(_prob,CubatureJLh())
# #             prob = QuadratureProblem(batch_iip_f(integrands[i]),ones(dim),3ones(dim),batch=10)
# #             for alg in multidim_batch_algs
# #                 if dim == 1 && (alg isa CubaDivonne || alg isa CubaCuhre)
# #                     continue
# #                 end
# #                 @info "Dimension = $dim, Alg = $alg, Integrand = $i"
# #                 if alg isa VEGAS
# #                     @test_broken sol = solve(prob,alg,reltol=1e-3,abstol=1e-3).retcode == :Success
# #                 else
# #                     sol = solve(prob,alg,reltol=1e-3,abstol=1e-3)
# #                     @test sol.u ≈ _sol.u rtol = 1e-2
# #                 end
# #             end
# #         end
# #     end
# # end
