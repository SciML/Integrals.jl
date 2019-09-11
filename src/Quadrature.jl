module Quadrature

using Requires, Reexport,  MonteCarloIntegration, QuadGK, HCubature
@reexport using DiffEqBase

struct QuadGKJL <: DiffEqBase.AbstractQuadratureAlgorithm end
struct HCubatureJL <: DiffEqBase.AbstractQuadratureAlgorithm end
struct VEGAS <: DiffEqBase.AbstractQuadratureAlgorithm
    nbins::Int
    ncalls::Int
end

VEGAS(;nbins = 100,ncalls = 1000) = VEGAS(nbins,ncalls)

abstract type AbstractCubaAlgorithm <: DiffEqBase.AbstractQuadratureAlgorithm end
struct CubaVegas <:AbstractCubaAlgorithm end
struct CubaSUAVE <: AbstractCubaAlgorithm end
struct CubaDivonne <: AbstractCubaAlgorithm end
struct CubaCuhre <: AbstractCubaAlgorithm end

abstract type AbstractCubatureJLAlgorithm <: DiffEqBase.AbstractQuadratureAlgorithm end
struct CubatureJLh <: AbstractCubatureJLAlgorithm end
struct CubatureJLp <: AbstractCubatureJLAlgorithm end

function scale_x!(_x,ub,lb,x)
    _x .= (ub .- lb) .* x .+ lb
    _x
end

function DiffEqBase.solve(prob::QuadratureProblem,::Nothing,args...;
                          reltol = 1e-8, abstol = 1e-8, kwargs...)
    if prob.lb isa Number
        solve(prob,QuadGKJL();reltol=reltol,abstol=abstol,kwargs...)
    elseif length(prob.lb) > 8 && reltol < 1e-4 || abstol < 1e-4
        solve(prob,VEGAS();reltol=reltol,abstol=abstol,kwargs...)
    else
        solve(prob,HCubatureJL();reltol=reltol,abstol=abstol,kwargs...)
    end
end

function DiffEqBase.solve(prob::QuadratureProblem,::QuadGKJL,args...;
                          reltol = 1e-8, abstol = 1e-8,
                          maxiters = typemax(Int),
                          kwargs...)
    if isinplace(prob) || prob.lb isa AbstractArray || prob.ub isa AbstractArray
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end
    @assert prob.batch == 0
    @assert prob.nout == 1
    p = prob.p
    f = x -> prob.f(x,p)
    val,err = quadgk(f, prob.lb, prob.ub,
                     rtol=reltol, atol=abstol,
                     kwargs...)
    DiffEqBase.build_solution(prob,QuadGKJL(),val,err,retcode = :Success)
end

function DiffEqBase.solve(prob::QuadratureProblem,::HCubatureJL,args...;
                          reltol = 1e-8, abstol = 1e-8,
                          maxiters = typemax(Int),
                          kwargs...)
    p = prob.p
    if isinplace(prob)
        dx = zeros(1)
        f = (x) -> (prob.f(dx,x,p); dx[1])
    else
        f = (x) -> prob.f(x,p)
    end
    @assert prob.batch == 0
    @assert prob.nout == 1
    if prob.lb isa Number
        val,err = hquadrature(f, prob.lb, prob.ub;
                            rtol=reltol, atol=abstol,
                            maxevals=maxiters, initdiv=1)
    else
        val,err = hcubature(f, prob.lb, prob.ub;
                            rtol=reltol, atol=abstol,
                            maxevals=maxiters, initdiv=1)
    end
    DiffEqBase.build_solution(prob,HCubatureJL(),val,err,retcode = :Success)
end

function DiffEqBase.solve(prob::QuadratureProblem,alg::VEGAS,args...;
                          reltol = 1e-8, abstol = 1e-8,
                          maxiters = typemax(Int),
                          kwargs...)
    p = prob.p
    @assert prob.nout == 1
    if prob.batch == 0
        if isinplace(prob)
          dx = zeros(1)
          f = (x) -> (prob.f(dx,x,p); dx[1])
        else
          f = (x) -> prob.f(x,p)
        end
    else
        if isinplace(prob)
          dx = zeros(prob.batch)
          f = (x) -> (prob.f(dx,x',p); dx)
        else
          f = (x) -> prob.f(x',p)
        end
    end
    val,err,chi = vegas(f, prob.lb, prob.ub, rtol=reltol, atol=abstol,
                        maxiter = maxiters, nbins = alg.nbins,
                        ncalls = alg.ncalls, batch=prob.batch != 0, kwargs...)
    DiffEqBase.build_solution(prob,alg,val,err,chi=chi,retcode = :Success)
end

function __init__()
    @require Cubature="667455a9-e2ce-5579-9412-b964f529a492" begin
        function DiffEqBase.solve(prob::QuadratureProblem,
                                  alg::AbstractCubatureJLAlgorithm, args...;
                                  reltol = 1e-8, abstol = 1e-8,
                                  maxiters = typemax(Int),
                                  kwargs...)
            nout = prob.nout
            if nout == 1
                if prob.batch == 0
                    if isinplace(prob)
                        dx = zeros(1)
                        f = (x) -> (prob.f(dx,x,prob.p); dx[1])
                    else
                        f = (x) -> prob.f(x,prob.p)
                    end
                    if prob.lb isa Number
                        if alg isa CubatureJLh
                            val,err = Cubature.hquadrature(f, prob.lb, prob.ub;
                                                           reltol=reltol, abstol=abstol,
                                                           maxevals=maxiters)
                        else
                            val,err = Cubature.pquadrature(f, prob.lb, prob.ub;
                                                           reltol=reltol, abstol=abstol,
                                                           maxevals=maxiters)
                        end
                    else
                        if alg isa CubatureJLh
                            val,err = Cubature.hcubature(f, prob.lb, prob.ub;
                                                         reltol=reltol, abstol=abstol,
                                                         maxevals=maxiters)
                        else
                            val,err = Cubature.pcubature(f, prob.lb, prob.ub;
                                                         reltol=reltol, abstol=abstol,
                                                         maxevals=maxiters)
                        end
                     end
                else
                    if isinplace(prob)
                        f = (x,dx) -> prob.f(dx,x,prob.p)
                    else
                        f = (x,dx) -> (dx .= prob.f(x,prob.p))
                    end
                    if prob.lb isa Number
                        if alg isa CubatureJLh
                            val,err = Cubature.hquadrature_v(f, prob.lb, prob.ub;
                                                             reltol=reltol, abstol=abstol,
                                                             maxevals=maxiters)
                        else
                            val,err = Cubature.pquadrature_v(f, prob.lb, prob.ub;
                                                             reltol=reltol, abstol=abstol,
                                                             maxevals=maxiters)
                        end
                    else
                        if alg isa CubatureJLh
                            val,err = Cubature.hcubature_v(f, prob.lb, prob.ub;
                                                           reltol=reltol, abstol=abstol,
                                                           maxevals=maxiters)
                        else
                            val,err = Cubature.pcubature_v(f, prob.lb, prob.ub;
                                                           reltol=reltol, abstol=abstol,
                                                           maxevals=maxiters)
                        end
                     end
                 end
             else
                 if prob.batch == 0
                     if isinplace(prob)
                         dx = similar(a)
                         f = (x,dx) -> (prob.f(dx,x,p); dx)
                     else
                         f = (x,dx) -> (dx .= prob.f(x,p))
                     end
                     if prob.lb isa Number
                         if alg isa CubatureJLh
                             val,err = Cubature.hquadrature(nout, f, prob.lb, prob.ub;
                                                            reltol=reltol, abstol=abstol,
                                                            maxevals=maxiters)
                         else
                             val,err = Cubature.pquadrature(nout, f, prob.lb, prob.ub;
                                                            reltol=reltol, abstol=abstol,
                                                            maxevals=maxiters)
                         end
                     else
                         if alg isa CubatureJLh
                             val,err = Cubature.hcubature(nout, f, prob.lb, prob.ub;
                                                          reltol=reltol, abstol=abstol,
                                                          maxevals=maxiters)
                         else
                             val,err = Cubature.pcubature(nout, f, prob.lb, prob.ub;
                                                          reltol=reltol, abstol=abstol,
                                                          maxevals=maxiters)
                         end
                      end
                 else
                     if isinplace(prob)
                         f = (x,dx) -> prob.f(dx,x,p)
                     else
                         f = (x,dx) -> (dx .= prob.f(x,p))
                     end
                     if prob.lb isa Number
                         if alg isa CubatureJLh
                             val,err = Cubature.hquadrature_v(nout, f, prob.lb, prob.ub;
                                                              reltol=reltol, abstol=abstol,
                                                              maxevals=maxiters)
                         else
                             val,err = Cubature.pquadrature_v(nout, f, prob.lb, prob.ub;
                                                              reltol=reltol, abstol=abstol,
                                                              maxevals=maxiters)
                         end
                     else
                         if alg isa CubatureJLh
                             val,err = Cubature.hcubature_v(nout, f, prob.lb, prob.ub;
                                                            reltol=reltol, abstol=abstol,
                                                            maxevals=maxiters)
                         else
                             val,err = Cubature.pcubature_v(nout, f, prob.lb, prob.ub;
                                                            reltol=reltol, abstol=abstol,
                                                            maxevals=maxiters)
                         end
                      end
                  end
              end
              DiffEqBase.build_solution(prob,alg,val,err,retcode = :Success)
        end
    end

    @require Cuba="8a292aeb-7a57-582c-b821-06e4c11590b1" begin
        function DiffEqBase.solve(prob::QuadratureProblem,alg::AbstractCubaAlgorithm,
                                  args...;
                                  reltol = 1e-8, abstol = 1e-8,
                                  maxiters = typemax(Int),
                                  kwargs...)
          p = prob.p
          if prob.lb isa Number && prob.batch == 0
              _x = Float64[prob.lb]
          elseif prob.lb isa Number
              _x = zeros(prob.batch)
          elseif prob.batch == 0
              _x = zeros(length(prob.lb))
          else
              _x = zeros(length(prob.lb),prob.batch)
          end
          ub = prob.ub
          lb = prob.lb

          if prob.batch == 0
              if isinplace(prob)
                  f = function (x,dx)
                      prob.f(dx,scale_x!(_x,ub,lb,x),p)
                      dx .*= prod((y)->y[1]-y[2],zip(ub,lb))
                  end
              else
                  f = function (x,dx)
                      dx .= prob.f(scale_x!(_x,ub,lb,x),p) .* prod((y)->y[1]-y[2],zip(ub,lb))
                  end
              end
          else
              if prob.lb isa Number
                  if isinplace(prob)
                      f = function (x,dx)
                          prob.f(dx',scale_x!(view(_x,1:length(x)),ub,lb,x),p)
                          dx .*= prod((y)->y[1]-y[2],zip(ub,lb))
                      end
                  else
                      f = function (x,dx)
                          dx .= prob.f(scale_x!(view(_x,1:length(x))',ub,lb,x),p)' .* prod((y)->y[1]-y[2],zip(ub,lb))
                      end
                  end
              else
                  if isinplace(prob)
                      f = function (x,dx)
                          prob.f(dx',scale_x!(view(_x,1:size(x,1),1:size(x,2)),ub,lb,x),p)
                          dx .*= prod((y)->y[1]-y[2],zip(ub,lb))
                      end
                  else
                      f = function (x,dx)
                          dx .= prob.f(scale_x!(view(_x,1:size(x,1),1:size(x,2)),ub,lb,x),p)' .* prod((y)->y[1]-y[2],zip(ub,lb))
                      end
                  end
              end
          end

          ndim = length(prob.lb)

          nvec = prob.batch == 0 ? 1 : prob.batch

          if alg isa CubaVegas
              out = Cuba.vegas(f, ndim, prob.nout; rtol = reltol, atol = abstol, nvec = nvec, kwargs...)
          elseif alg isa CubaSUAVE
              out = Cuba.suave(f, ndim, prob.nout; rtol = reltol, atol = abstol, nvec = nvec, kwargs...)
          elseif alg isa CubaDivonne
              out = Cuba.divonne(f, ndim, prob.nout; rtol = reltol, atol = abstol, nvec = nvec, kwargs...)
          elseif alg isa CubaCuhre
              out = Cuba.cuhre(f, ndim, prob.nout; rtol = reltol, atol = abstol, nvec = nvec, kwargs...)
          end

          if prob.nout == 1
              val = out.integral[1]
          else
              val = out.integral
          end

          DiffEqBase.build_solution(prob,alg,val,out.error,
                         chi=out.probability,retcode = :Success)
        end
    end
end

export QuadGKJL, HCubatureJL, VEGAS
export CubatureJLh, CubatureJLp
export CubaVegas, CubaSUAVE, CubaDivonne, CubaCuhre
end # module
