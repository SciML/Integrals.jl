module IntegralsCubatureExt

using Integrals, Cubature

using Integrals: scale_x, scale_x!, CubatureJLh, CubatureJLp, AbstractCubatureJLAlgorithm
using SciMLLogging: @SciMLMessage

function Integrals.__solvebp_call(
        prob::IntegralProblem,
        alg::AbstractCubatureJLAlgorithm,
        sensealg, domain, p;
        reltol = 1.0e-8, abstol = 1.0e-8,
        maxiters = typemax(Int),
        verbose = Integrals.DEFAULT_VERBOSE
    )
    lb, ub = domain
    mid = (lb + ub) / 2

    @SciMLMessage(
        lazy"$(typeof(alg).name.name): starting $(mid isa Number ? \"1D\" : \"multi-dimensional\") integration with reltol=$reltol, abstol=$abstol",
        verbose, :algorithm_selection
    )

    # we get to pick fdim or not based on the IntegralFunction and its output dimensions
    f = prob.f
    prototype = Integrals.get_prototype(prob)

    @assert eltype(prototype) <: Real "Cubature.jl is only compatible with real-valued integrands"

    if f isa BatchIntegralFunction
        @SciMLMessage("Using batch evaluation mode", verbose, :batch_mode)

        if prototype isa AbstractVector # this branch could be omitted since the following one should work similarly
            if isinplace(f)
                @SciMLMessage("Using in-place evaluation", verbose, :batch_mode)
                # dx is a Vector, but we provide the integrand a vector of the same type as
                # y, which needs to be resized since the number of batch points changes.
                _f = let y = similar(prototype)
                    (u, v) -> begin
                        resize!(y, length(v))
                        f(y, u, p)
                        v .= y
                    end
                end
            else
                _f = (u, v) -> (v .= f(u, p))
            end
            if mid isa Number
                if alg isa CubatureJLh
                    val,
                        err = Cubature.hquadrature_v(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                else
                    val,
                        err = Cubature.pquadrature_v(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                end
            else
                if alg isa CubatureJLh
                    val,
                        err = Cubature.hcubature_v(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                else
                    val,
                        err = Cubature.pcubature_v(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                end
            end
        elseif prototype isa AbstractArray
            fsize = size(prototype)[begin:(end - 1)]
            fdim = prod(fsize)
            if isinplace(f)
                @SciMLMessage("Using in-place evaluation", verbose, :batch_mode)
                # dx is a Matrix, but to provide a buffer of the same type as y, we make
                # would like to make views of a larger buffer, but CubatureJL doesn't set
                # a hard limit for max_batch, so we allocate a new buffer with the needed size
                _f = let fsize = fsize
                    (u, v) -> begin
                        y = similar(prototype, fsize..., size(v, 2))
                        f(y, u, p)
                        v .= reshape(y, fdim, size(v, 2))
                    end
                end
            else
                _f = (u, v) -> (v .= reshape(f(u, p), fdim, size(v, 2)))
            end
            if mid isa Number
                if alg isa CubatureJLh
                    val_,
                        err = Cubature.hquadrature_v(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                else
                    val_,
                        err = Cubature.pquadrature_v(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                end
            else
                if alg isa CubatureJLh
                    val_,
                        err = Cubature.hcubature_v(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                else
                    val_,
                        err = Cubature.pcubature_v(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                end
            end
            val = reshape(val_, fsize...)
        else
            error("BatchIntegralFunction integrands must be arrays for Cubature.jl")
        end
    else
        if prototype isa Real
            # no inplace in this case, since the integrand_prototype would be mutable
            _f = u -> f(u, p)
            if lb isa Number
                if alg isa CubatureJLh
                    val,
                        err = Cubature.hquadrature(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                else
                    val,
                        err = Cubature.pquadrature(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                end
            else
                if alg isa CubatureJLh
                    val,
                        err = Cubature.hcubature(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                else
                    val,
                        err = Cubature.pcubature(
                        _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters
                    )
                end
            end
        elseif prototype isa AbstractArray
            fsize = size(prototype)
            fdim = length(prototype)
            if isinplace(prob)
                _f = let y = similar(prototype)
                    (u, v) -> (f(y, u, p); v .= vec(y))
                end
            else
                _f = (u, v) -> (v .= vec(f(u, p)))
            end
            if mid isa Number
                if alg isa CubatureJLh
                    val_,
                        err = Cubature.hquadrature(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                else
                    val_,
                        err = Cubature.pquadrature(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                end
            else
                if alg isa CubatureJLh
                    val_,
                        err = Cubature.hcubature(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                else
                    val_,
                        err = Cubature.pcubature(
                        fdim, _f, lb, ub;
                        reltol = reltol, abstol = abstol,
                        maxevals = maxiters, error_norm = alg.error_norm
                    )
                end
            end
            val = reshape(val_, fsize)
        else
            error("IntegralFunctions must be scalars or arrays for Cubature.jl")
        end
    end

    @SciMLMessage(
        lazy"$(typeof(alg).name.name) converged: val=$val, err=$err",
        verbose, :convergence_result
    )

    return SciMLBase.build_solution(prob, alg, val, err, retcode = ReturnCode.Success)
end

end
