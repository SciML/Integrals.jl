##
using Quadrature
using Cubature, Cuba
using Test

max_dim_test = 2
max_nout_test = 2
reltol=1e-3
abstol=1e-3

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

## remove nout and make size(x)
integrands_v = [
                (x,p; nout=2) -> collect(1:nout)
                (x,p; nout=2) -> integrands[2](x,p)*collect(1:nout)
                ]
iip_integrands_v = [ (dx,x,p; nout=2)-> (dx .= f(x,p,nout=nout)) for f ∈ integrands_v]

exact_sol = [
                (ndim, nout, lb, ub) -> prod(ub-lb),
                (ndim, nout, lb, ub) -> prod(sin.(ub)-sin.(lb))
            ]

exact_sol_v = [
                (ndim, nout, lb, ub) -> prod(ub-lb) * collect(1:nout),
                (ndim, nout, lb, ub) -> exact_sol[2](ndim,nout,lb,ub) * collect(1:nout)
            ]

batch_f(f) = (pts,p) -> begin
  fevals = zeros(size(pts,2))
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

# alg = CubatureJLh()
alg = CubaSUAVE()
# alg = HCubatureJL()
ndim = 1
nout = 1
# lb,ub = ([0.0],[1.0])
lb,ub = (1,3)
i = 1
batch = 10


# f = (x,p) -> begin @show x; integrands_v[i](x,p,nout=nout); end
# f = (x,p) -> iip_integrands_v[i](x,p,nout=nout)
# f = (dx,x,p) ->iip_integrands_v[i](dx,x,p,nout=nout)
f = batch_f(integrands[i])
x = [3.0]; dx = similar(x); f(x,1)
prob = QuadratureProblem(f,lb,ub, nout=1,batch=batch)
@show v1 = solve(prob,alg,reltol=1e-4,abstol=1e-4).u
# Juno.@enter solve(prob,alg,reltol=1e-4,abstol=1e-4)
# @show v2 = hcubature(nout, (x,v) -> f(v,x,1.0) , lb, ub, reltol = reltol, abstol=abstol)[1]
# @show v2 = Cubature.hquadrature(nout, (x,v) -> f(v,x,1.0) , lb, ub, reltol = reltol, abstol=abstol)[1]
# @show v3 = suave((x,v) -> v .= integrands_v[i](x,1.0,nout=nout) , ndim, nout,rtol = reltol, atol=1e-3)[1]
# v1≈v2
[exact_sol[i](ndim,1,lb,ub)]

##
@testset "Standard Single Dimension Integrands" begin
    lb,ub = (1,3)
    for i in 1:length(integrands)
        prob = QuadratureProblem(integrands[i],lb,ub)
        for alg in algs
            if alg_req[alg].min_dim > 1
                continue
            end
            @info "Dimension = 1, Alg = $alg, Integrand = $i"
            sol = solve(prob,alg,reltol=reltol,abstol=abstol)
            @test sol.u ≈ exact_sol[i](1,1,lb,ub) rtol = 1e-2
        end
    end
end

@testset "Standard Integrands" begin
    for i in 1:length(integrands)
        for dim = 1:max_dim_test
            lb, ub = (ones(dim), 3ones(dim))
            prob = QuadratureProblem(integrands[i],lb,ub)
            for alg in algs
                req = alg_req[alg]
                if dim > req.max_dim || dim < req.min_dim || alg isa QuadGKJL  #QuadGKJL requires numbers, not single element arrays
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=reltol,abstol=abstol)
                @test sol.u ≈ exact_sol[i](dim,1,lb,ub) rtol = 1e-2
            end
        end
    end
end

@testset "In-place Standard Integrands" begin
    for i in 1:length(iip_integrands)
        for dim = 1:max_dim_test
            lb, ub = (ones(dim), 3ones(dim))
            prob = QuadratureProblem(iip_integrands[i],lb,ub)
            _sol = solve(prob,CubatureJLh())
            for alg in algs
                req = alg_req[alg]
                if dim > req.max_dim || dim < req.min_dim || alg isa QuadGKJL  #QuadGKJL requires numbers, not single element arrays
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                if alg isa HCubatureJL  && dim == 1 # HCubature library requires finer tol to pass test. When requiring array outputs for iip integrands
                    sol = solve(prob,alg,reltol=1e-5,abstol=1e-5)
                else
                    sol = solve(prob,alg,reltol=reltol,abstol=abstol)
                end
                @test sol.u ≈ [exact_sol[i](dim,1,lb,ub)] rtol = 1e-2
            end
        end
    end
end

@testset "Batched Single Dimension Integrands" begin
    (lb,ub) = (1,3)
    (dim, nout) = (1,1)
    for i in 1:length(integrands)
        prob = QuadratureProblem(batch_f(integrands[i]),lb,ub,batch=10)
        for alg in algs
            req = alg_req[alg]
            if req.min_dim > 1 || !req.allows_batch
                continue
            end
            @info "Dimension = 1, Alg = $alg, Integrand = $i"
            sol = solve(prob,alg,reltol=reltol,abstol=abstol)
            @test sol.u ≈ [exact_sol[i](dim,nout,lb,ub)] rtol = 1e-2
        end
    end
end

@testset "Batched Standard Integrands" begin
    nout = 1
    for i in 1:length(integrands)
        for dim = 1:max_dim_test
            (lb,ub) = (ones(dim),3ones(dim))
            prob = QuadratureProblem(batch_f(integrands[i]),lb,ub,batch=10)
            for alg in algs
                req = alg_req[alg]
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=reltol,abstol=abstol)
                @test sol.u ≈ [exact_sol[i](dim,nout,lb,ub)] rtol = 1e-2
            end
        end
    end
end

@testset "In-Place Batched Standard Integrands" begin
    nout = 1
    for i in 1:length(iip_integrands)
        for dim = 1:max_dim_test
            (lb,ub) = (ones(dim),3ones(dim))
            prob = QuadratureProblem(batch_iip_f(integrands[i]),lb,ub,batch=10)
            for alg in algs
                req = req = alg_req[alg]
                if dim > req.max_dim || dim < req.min_dim || !req.allows_batch || !req.allows_iip
                    continue
                end
                @info "Dimension = $dim, Alg = $alg, Integrand = $i"
                sol = solve(prob,alg,reltol=reltol,abstol=abstol)
                @test sol.u ≈ [exact_sol[i](dim,nout,lb,ub)] rtol = 1e-2
            end
        end
    end
end

######## Vector Valued Integrands
@testset "Standard Single Dimension Vector Integrands" begin
    lb,ub = (1,3)
    for i in 1:length(integrands_v)
        for nout = 1:max_nout_test
            prob = QuadratureProblem((x,p) -> integrands_v[i](x,p,nout=nout),lb,ub, nout = nout)
            for alg in algs
                req = alg_req[alg]
                if req.min_dim > 1 || req.nout < nout
                    continue
                end
                @info "Dimension = 1, Alg = $alg, Integrand = $i, Output Dimension = $nout"
                sol = solve(prob,alg,reltol=reltol,abstol=abstol)
                @test sol.u ≈ exact_sol_v[i](1,nout,lb,ub) rtol = 1e-2
            end
        end
    end
end

@testset "Standard Vector Integrands" begin
    for i in 1:length(integrands_v)
        for dim = 1:max_dim_test
            lb, ub = (ones(dim), 3ones(dim))
            for nout = 1:max_nout_test
                prob = QuadratureProblem((x,p) -> integrands_v[i](x,p,nout=nout),lb,ub, nout = nout)
                for alg in algs
                    req = alg_req[alg]
                    if dim > req.max_dim || dim < req.min_dim || req.nout < nout || alg isa QuadGKJL  #QuadGKJL requires numbers, not single element arrays
                        continue
                    end
                    @info "Dimension = 1, Alg = $alg, Integrand = $i, Output Dimension = $nout"
                    sol = solve(prob,alg,reltol=reltol,abstol=abstol)
                    @test sol.u ≈ exact_sol_v[i](dim,nout,lb,ub) rtol = 1e-2
                end
            end
        end
    end
end

@testset "In-place Standard Vector Integrands" begin
    for i in 1:length(iip_integrands_v)
        for dim = 1:max_dim_test
            lb, ub = (ones(dim), 3ones(dim))
            for nout = 1:max_nout_test
                prob = QuadratureProblem((dx,x,p) ->iip_integrands_v[i](dx,x,p,nout=nout),lb,ub,nout = nout)
                _sol = solve(prob,CubatureJLh())
                for alg in algs
                    req = alg_req[alg]
                    if dim > req.max_dim || dim < req.min_dim || req.nout < nout || alg isa QuadGKJL  #QuadGKJL requires numbers, not single element arrays
                        continue
                    end
                    @info "Dimension = 1, Alg = $alg, Integrand = $i, Output Dimension = $nout"
                    sol = solve(prob,alg,reltol=reltol,abstol=abstol)
                    @test sol.u ≈ exact_sol_v[i](dim,nout,lb,ub) rtol = 1e-2
                end
            end
        end
    end
end
#
# @testset "Batched Single Dimension Vector Integrands" begin
#     (lb,ub) = (1,3)
#     (dim, nout) = (1,1)
#     for i in 1:length(integrands_v)
#         prob = QuadratureProblem(batch_f(integrands_v[i]),lb,ub,batch=10,nout = nout)
#         for alg in algs
#             req = alg_req[alg]
#             if req.min_dim > 1 || !req.allows_batch || req.nout < nout
#                 continue
#             end
#             @info "Dimension = 1, Alg = $alg, Integrand = $i, Output Dimension = $nout"
#             sol = solve(prob,alg,reltol=reltol,abstol=abstol)
#             @test sol.u ≈ exact_sol_v[i](dim,nout,lb,ub) rtol = 1e-2
#         end
#     end
# end
#
# @testset "Batched Standard Vector Integrands" begin
#     nout = 1
#     for i in 1:length(integrands_v)
#         for dim = 1:5
#             (lb,ub) = (ones(dim),3ones(dim))
#             prob = QuadratureProblem(batch_f(integrands_v[i]),lb,ub,batch=10,nout = nout)
#             for alg in algs
#                 req = alg_req[alg]
#                 if dim > req.max_dim || dim < req.min_dim || !req.allows_batch || req.nout < nout
#                     continue
#                 end
#                 @info "Dimension = 1, Alg = $alg, Integrand = $i, Output Dimension = $nout"
#                 sol = solve(prob,alg,reltol=reltol,abstol=abstol)
#                 @test sol.u ≈ exact_sol_v[i](dim,nout,lb,ub) rtol = 1e-2
#             end
#         end
#     end
# end
#
# @testset "In-Place Batched Standard Vector Integrands" begin
#     nout = 1
#     for i in 1:length(iip_integrands_v)
#         for dim = 1:5
#             (lb,ub) = (ones(dim),3ones(dim))
#             prob = QuadratureProblem(batch_iip_f(integrands_v[i]),lb,ub,batch=10,nout = nout)
#             for alg in algs
#                 req = req = alg_req[alg]
#                 if dim > req.max_dim || dim < req.min_dim || !req.allows_batch || !req.allows_iip || req.nout < nout
#                     continue
#                 end
#                 @info "Dimension = 1, Alg = $alg, Integrand = $i, Output Dimension = $nout"
#                 sol = solve(prob,alg,reltol=reltol,abstol=abstol)
#                 @test sol.u ≈ exact_sol_v[i](dim,nout,lb,ub) rtol = 1e-2
#             end
#         end
#     end
# end
