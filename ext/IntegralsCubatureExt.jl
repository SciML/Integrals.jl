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
    y = if prob.f isa BatchIntegralFunction
        isinplace(prob.f) ? prob.f.integrand_prototype :
        mid isa Number ? prob.f(eltype(mid)[], p) :
        prob.f(Matrix{eltype(mid)}(undef, length(mid), 0), p)
    else
        # we evaluate the oop function to decide whether the output should be vectorized
        isinplace(prob.f) ? prob.f.integrand_prototype : prob.f(mid, p)
    end

    @assert eltype(y)<:Real "Cubature.jl is only compatible with real-valued integrands"

    if prob.f isa BatchIntegralFunction
        if y isa AbstractVector # this branch could be omitted since the following one should work similarly
            if isinplace(prob)
                # dx is a Vector, but we provide the integrand a vector of the same type as
                # y, which needs to be resized since the number of batch points changes.
                dy = similar(y)
                f = (x, dx) -> begin
                    resize!(dy, length(dx))
                    prob.f(dy, x, p)
                    dx .= dy
                end
            else
                f = (x, dx) -> (dx .= prob.f(x, p))
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
        elseif y isa AbstractArray
            bfsize = size(y)[begin:(end - 1)]
            bfdim = prod(bfsize)
            if isinplace(prob)
                # dx is a Matrix, but to provide a buffer of the same type as y, we make
                # would like to make views of a larger buffer, but CubatureJL doesn't set
                # a hard limit for max_batch, so we allocate a new buffer with the needed size
                f = (x, dx) -> begin
                    dy = similar(y, bfsize..., size(dx, 2))
                    prob.f(dy, x, p)
                    dx .= reshape(dy, bfdim, size(dx, 2))
                end
            else
                f = (x, dx) -> (dx .= reshape(prob.f(x, p), bfdim, size(dx, 2)))
            end
            if mid isa Number
                if alg isa CubatureJLh
                    val_, err = Cubature.hquadrature_v(bfdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                else
                    val_, err = Cubature.pquadrature_v(bfdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                end
            else
                if alg isa CubatureJLh
                    val_, err = Cubature.hcubature_v(bfdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                else
                    val_, err = Cubature.pcubature_v(bfdim, f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm)
                end
            end
            val = reshape(val_, bfsize...)
        else
            error("BatchIntegralFunction integrands must be arrays for Cubature.jl")
        end
    else
        if y isa Real
            # no inplace in this case, since the integrand_prototype would be mutable
            f = x -> prob.f(x, p)
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
        elseif y isa AbstractArray
            fsize = size(y)
            fdim = length(y)
            if isinplace(prob)
                dy = similar(y)
                f = (x, v) -> (prob.f(dy, x, p); v .= vec(dy))
            else
                f = (x, v) -> (v .= vec(prob.f(x, p)))
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
