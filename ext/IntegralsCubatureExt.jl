module IntegralsCubatureExt

using Integrals, Cubature

using Integrals: scale_x, scale_x!, CubatureJLh, CubatureJLp, AbstractCubatureJLAlgorithm

function Integrals.__solvebp_call(prob::IntegralProblem,
        alg::AbstractCubatureJLAlgorithm,
        sensealg, domain, p;
        reltol = 1e-8, abstol = 1e-8,
        maxiters = typemax(Int))
    lb, ub = domain
    mid = (lb + ub) / 2

    # we get to pick fdim or not based on the IntegralFunction and its output dimensions
    prototype = if prob.f isa BatchIntegralFunction
        isinplace(prob.f) ? prob.f.integrand_prototype :
        mid isa Number ? prob.f(eltype(mid)[], p) :
        prob.f(Matrix{eltype(mid)}(undef, length(mid), 0), p)
    else
        # we evaluate the oop function to decide whether the output should be vectorized
        isinplace(prob.f) ? prob.f.integrand_prototype : prob.f(mid, p)
    end

    @assert eltype(prototype)<:Real "Cubature.jl is only compatible with real-valued integrands"

    if prob.f isa BatchIntegralFunction
        if prototype isa AbstractVector # this branch could be omitted since the following one should work similarly
            if isinplace(prob)
                # dx is a Vector, but we provide the integrand a vector of the same type as
                # y, which needs to be resized since the number of batch points changes.
                f = let y = similar(prototype)
                    (u, v) -> begin
                        resize!(y, length(v))
                        prob.f(y, u, p)
                        v .= y
                    end
                end
            else
                f = (u, v) -> (v .= prob.f(u, p))
            end
            if mid isa Number
                if alg isa CubatureJLh
                    val, err = Cubature.hquadrature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pquadrature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pcubature_v(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            end
        elseif prototype isa AbstractArray
            fsize = size(prototype)[begin:(end - 1)]
            fdim = prod(fsize)
            if isinplace(prob)
                # dx is a Matrix, but to provide a buffer of the same type as y, we make
                # would like to make views of a larger buffer, but CubatureJL doesn't set
                # a hard limit for max_batch, so we allocate a new buffer with the needed size
                f = let fsize = fsize
                    (u, v) -> begin
                        y = similar(prototype, fsize..., size(v, 2))
                        prob.f(y, u, p)
                        v .= reshape(y, fdim, size(v, 2))
                    end
                end
            else
                f = (u, v) -> (v .= reshape(prob.f(u, p), fdim, size(v, 2)))
            end
            if mid isa Number
                if alg isa CubatureJLh
                    val_, err = Cubature.hquadrature_v(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                else
                    val_, err = Cubature.pquadrature_v(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                end
            else
                if alg isa CubatureJLh
                    val_, err = Cubature.hcubature_v(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                else
                    val_, err = Cubature.pcubature_v(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                end
            end
            val = reshape(val_, fsize...)
        else
            error("BatchIntegralFunction integrands must be arrays for Cubature.jl")
        end
    else
        if prototype isa Real
            # no inplace in this case, since the integrand_prototype would be mutable
            f = u -> prob.f(u, p)
            if lb isa Number
                if alg isa CubatureJLh
                    val, err = Cubature.hquadrature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pquadrature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            else
                if alg isa CubatureJLh
                    val, err = Cubature.hcubature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                else
                    val, err = Cubature.pcubature(f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters)
                end
            end
        elseif prototype isa AbstractArray
            fsize = size(prototype)
            fdim = length(prototype)
            if isinplace(prob)
                f = let y = similar(prototype)
                    (u, v) -> (prob.f(y, u, p); v .= vec(y))
                end
            else
                f = (u, v) -> (v .= vec(prob.f(u, p)))
            end
            if mid isa Number
                if alg isa CubatureJLh
                    val_, err = Cubature.hquadrature(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                else
                    val_, err = Cubature.pquadrature(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                end
            else
                if alg isa CubatureJLh
                    val_, err = Cubature.hcubature(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                else
                    val_, err = Cubature.pcubature(fdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                end
            end
            val = reshape(val_, fsize)
        else
            error("IntegralFunctions must be scalars or arrays for Cubature.jl")
        end
    end
    SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

end
