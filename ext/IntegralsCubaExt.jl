module IntegralsCubaExt

using Integrals, Cuba
import Integrals: transformation_if_inf,
    scale_x, scale_x!, CubaVegas, AbstractCubaAlgorithm,
    CubaSUAVE, CubaDivonne, CubaCuhre

function Integrals.__solvebp_call(
        prob::IntegralProblem, alg::AbstractCubaAlgorithm,
        sensealg,
        domain, p;
        reltol = 1.0e-4, abstol = 1.0e-12,
        maxiters = 1000000
    )
    @assert maxiters >= 1000 "maxiters for $alg should be larger than 1000"
    lb, ub = domain
    mid = (lb + ub) / 2
    ndim = length(mid)
    (vol = prod(map(-, ub, lb))) isa Real ||
        throw(ArgumentError("Cuba.jl only supports real-valued integrands"))
    # we could support other types by multiplying by the jacobian determinant at the end

    f = prob.f
    prototype = Integrals.get_prototype(prob)
    if f isa BatchIntegralFunction
        fsize = size(prototype)[begin:(end - 1)]
        ncomp = prod(fsize)
        nvec = min(maxiters, f.max_batch)
        # nvec == 1 in Cuba will change vectors to matrices, so we won't support it when
        # batching
        nvec > 1 ||
            throw(ArgumentError("BatchIntegralFunction must take multiple batch points"))

        if mid isa Real
            _x = zeros(typeof(mid), nvec)
            scale = x -> scale_x!(resize!(_x, length(x)), ub, lb, vec(x))
        else
            _x = zeros(eltype(mid), length(mid), nvec)
            scale = x -> scale_x!(view(_x, :, 1:size(x, 2)), ub, lb, x)
        end

        if isinplace(f)
            ax = ntuple(_ -> (:), length(fsize))
            _f = let y_ = similar(prototype, fsize..., nvec)
                function (u, _y)
                    y = @view(y_[ax..., begin:(begin + size(_y, 2) - 1)])
                    f(y, scale(u), p)
                    return _y .= reshape(y, size(_y)) .* vol
                end
            end
        else
            _f = (u, y) -> y .= reshape(f(scale(u), p), size(y)) .* vol
        end
    else
        nvec = 1
        ncomp = length(prototype)

        if mid isa Real
            scale = x -> scale_x(ub, lb, only(x))
        else
            _x = zeros(eltype(mid), length(mid))
            scale = x -> scale_x!(_x, ub, lb, x)
        end

        if isinplace(f)
            _f = let y = similar(prototype)
                (u, _y) -> begin
                    f(y, scale(u), p)
                    _y .= vec(y) .* vol
                end
            end
        else
            _f = (u, y) -> y .= Iterators.flatten(f(scale(u), p)) .* vol
        end
    end

    out = if alg isa CubaVegas
        Cuba.vegas(
            _f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, seed = alg.seed, minevals = alg.minevals,
            nstart = alg.nstart, nincrease = alg.nincrease,
            gridno = alg.gridno
        )
    elseif alg isa CubaSUAVE
        Cuba.suave(
            _f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, seed = alg.seed, minevals = alg.minevals,
            nnew = alg.nnew, nmin = alg.nmin, flatness = alg.flatness
        )
    elseif alg isa CubaDivonne
        Cuba.divonne(
            _f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, seed = alg.seed, minevals = alg.minevals,
            key1 = alg.key1, key2 = alg.key2, key3 = alg.key3,
            maxpass = alg.maxpass, border = alg.border,
            maxchisq = alg.maxchisq, mindeviation = alg.mindeviation
        )
    elseif alg isa CubaCuhre
        Cuba.cuhre(
            _f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, minevals = alg.minevals, key = alg.key
        )
    end

    # out.integral is a Vector{Float64}, but we want to return it to the shape of the integrand
    val = if f isa BatchIntegralFunction
        if prototype isa AbstractVector
            out.integral[1]
        else
            reshape(out.integral, fsize)
        end
    else
        if prototype isa Real
            out.integral[1]
        elseif prototype isa AbstractVector
            out.integral
        else
            reshape(out.integral, size(prototype))
        end
    end

    return SciMLBase.build_solution(
        prob, alg, val, out.error,
        chi = out.probability, retcode = ReturnCode.Success
    )
end

end
