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


function DiffEqBase.solve(prob::QuadratureProblem,::Nothing;
                          reltol = 1e-8, abstol = 1e-8, kwargs...)
    if prob.lb isa Number
        solve(prob,QuadGKJL();reltol=reltol,abstol=abstol,kwargs...)
    elseif length(prob.lb) > 8 && reltol < 1e-4 || abstol < 1e-4
        solve(prob,VEGAS();reltol=reltol,abstol=abstol,kwargs...)
    else
        solve(prob,HCubatureJL();reltol=reltol,abstol=abstol,kwargs...)
    end
end

function DiffEqBase.solve(prob::QuadratureProblem,::QuadGKJL;
                          reltol = 1e-8, abstol = 1e-8,
                          maxiters = typemax(Int),
                          kwargs...)
    if isinplace(prob) || prob.lb isa AbstractArray || prob.ub isa AbstractArray
        error("QuadGKJL only accepts one-dimensional quadrature problems.")
    end
    @assert !prob.batch
    @assert prob.nout == 1
    f = x -> prob.f(x,p)
    val,err = QuadGK.quadgk(f, prob.lb, prob.ub,
                            rtol=reltol, atol=abstol,
                            kwargs...)
    build_solution(prob,QuadGKJL(),val,err,retcode = :Success)
end

function DiffEqBase.solve(prob::QuadratureProblem,::HCubatureJL;
                          reltol = 1e-8, abstol = 1e-8,
                          maxiters = typemax(Int),
                          kwargs...)
    if isinplace(prob)
        dx = similar(a)
        f = (x) -> (prob.f(dx,x,p); dx)
    else
        f = (x) -> prob.f(x,p)
    end
    @assert !prob.batch
    @assert prob.nout == 1
    val,err = hcubature(f, prob.lb, prob.ub;
                        rtol=reltol, atol=abstol,
                        maxevals=maxiters, initdiv=1)
    build_solution(prob,HCubatureJL(),val,err,retcode = :Success)
end

function DiffEqBase.solve(prob::QuadratureProblem,alg::VEGAS;
                          reltol = 1e-8, abstol = 1e-8,
                          maxiters = typemax(Int),
                          kwargs...)
    @assert prob.nout == 1
    if !prob.batch
        if isinplace(prob)
          dx = similar(a)
          f = (x) -> (prob.f(dx,x,p); dx)
        else
          f = (x) -> prob.f(x,p)
        end
    else
        if isinplace(prob)
          dx = similar(a)
          f = (x) -> (prob.f(dx,x',p); dx)
        else
          f = (x) -> prob.f(x',p)
        end
    end
    val,err,chi = vegas(f, prob.lb, prob.ub, rtol=reltol, atol=abstol,
                        maxiter = maxiters, nbins = alg.nbins,
                        ncalls = alg.ncalls, kwargs...)
    build_solution(prob,alg,val,err,chi=chi,retcode = :Success)
end

@require Cubature="667455a9-e2ce-5579-9412-b964f529a492" begin
    abstract type AbstractCubatureJLAlgorithm <: DiffEqBase.AbstractQuadratureAlgorithm end
    struct CubatureJLh <: AbstractCubatureJLAlgorithm end
    struct CubatureJLp <: AbstractCubatureJLAlgorithm end

    function DiffEqBase.solve(prob::QuadratureProblem,alg::AbstractCubatureJLAlgorithm;
                              reltol = 1e-8, abstol = 1e-8,
                              maxiters = typemax(Int),
                              kwargs...)
        nout = prob.nout
        if nout == 1
            if !prob.batch
                if isinplace(prob)
                    dx = similar(a)
                    f = (x) -> (prob.f(dx,x,p); dx)
                else
                    f = (x) -> prob.f(x,p)
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
                    f = (x,dx) -> prob.f(dx,x,p)
                else
                    f = (x,dx) -> (dx .= prob.f(x,p))
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
             if !prob.batch
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
          build_solution(prob,alg,val,err,retcode = :Success)
    end
    export CubatureJLh, CubatureJLp
end


@require Cuba="8a292aeb-7a57-582c-b821-06e4c11590b1" begin
    abstract type AbstractCubaAlgorithm <: DiffEqBase.AbstractQuadratureAlgorithm end
    struct CubaVegas <:AbstractCubaAlgorithm end
    struct CubaSUAVE <: AbstractCubaAlgorithm end
    struct CubaDivonne <: AbstractCubaAlgorithm end
    struct CubaCuhre <: AbstractCubaAlgorithm end

    function scale_x!(_x,ub,lb,x)
        _x .= (ub .- lb).*x .+ lb
        _x
    end

    function DiffEqBase.solve(prob::QuadratureProblem,alg::AbstractCubaAlgorithm;
                              reltol = 1e-8, abstol = 1e-8,
                              maxiters = typemax(Int),
                              kwargs...)
      _x = similar(x)
      ub = prob.ub
      lb = prob.lb

      if isinplace(prob)
          f = (x,dx) -> prob.f(dx,scale_x!(_x,ub,lb,x),p)
      else
          f = (x,dx) -> (dx .= prob.f(scale_x!(_x,ub,lb,x),p))
      end
      ndim = length(prob.lb)

      if alg isa CubaVegas
          out = vegas(f, ndim, prob.nout; rtol = reltol, atol = abstol, kwargs...)
      elseif alg isa CubaSUAVE
          out = suave(f, ndim, prob.nout; rtol = reltol, atol = abstol, kwargs...)
      elseif alg isa CubaDivonne
          out = divonne(f, ndim, prob.nout; rtol = reltol, atol = abstol, kwargs...)
      elseif alg isa CubaCuhre
          out = cuhre(f, ndim, prob.nout; rtol = reltol, atol = abstol, kwargs...)
      end

      if prob.nout == 1
          val = out.integral[1]
      else
          val = out.integral
      end

      build_solution(prob,alg,val,out.error,
                     chi=out.probl,retcode = :Success)
    end

    export CubaVegas, CubaSUAVE, CubaDivonne, CubaCuhre
end

export QuadGKJL, HCubatureJL, VEGAS

end # module
