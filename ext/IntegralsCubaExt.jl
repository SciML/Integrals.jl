module IntegralsCubaExt

using Integrals, Cuba
import Integrals: transformation_if_inf, scale_x, scale_x!

function Integrals.__solvebp_call(prob::IntegralProblem, alg::AbstractCubaAlgorithm,
        sensealg,
        domain, p;
        reltol = 1e-8, abstol = 1e-8,
        maxiters = alg isa CubaSUAVE ? 1000000 : typemax(Int))
    @assert maxiters>=1000 "maxiters for $alg should be larger than 1000"
    lb, ub = domain
    mid = (lb + ub) / 2
    ndim = length(mid)
    (vol = prod(map(-, ub, lb))) isa Real ||
        throw(ArgumentError("Cuba.jl only supports real-valued integrands"))
    # we could support other types by multiplying by the jacobian determinant at the end

    if prob.f isa BatchIntegralFunction
        # nvec == 1 in Cuba will change vectors to matrices, so we won't support it when
        # batching
        (nvec = prob.f.max_batch) > 1 ||
            throw(ArgumentError("BatchIntegralFunction must take multiple batch points"))

        if mid isa Real
            _x = zeros(typeof(mid), prob.f.max_batch)
            scale = x -> scale_x!(resize!(_x, length(x)), ub, lb, vec(x))
        else
            _x = zeros(eltype(mid), length(mid), prob.f.max_batch)
            scale = x -> scale_x!(view(_x, :, 1:size(x, 2)), ub, lb, x)
        end

        if isinplace(prob)
            fsize = size(prob.f.integrand_prototype)[begin:(end - 1)]
            y = similar(prob.f.integrand_prototype, fsize..., nvec)
            ax = map(_ -> (:), fsize)
            f = function (x, dx)
                dy = @view(y[ax..., begin:(begin + size(dx, 2) - 1)])
                prob.f(dy, scale(x), p)
                dx .= reshape(dy, :, size(dx, 2)) .* vol
            end
        else
            y = mid isa Number ? prob.f(typeof(mid)[], p) :
                prob.f(Matrix{typeof(mid)}(undef, length(mid), 0), p)
            fsize = size(y)[begin:(end - 1)]
            f = (x, dx) -> dx .= reshape(prob.f(scale(x), p), :, size(dx, 2)) .* vol
        end
        ncomp = prod(fsize)
    else
        nvec = 1

        if mid isa Real
            scale = x -> scale_x(ub, lb, only(x))
        else
            _x = zeros(eltype(mid), length(mid))
            scale = x -> scale_x!(_x, ub, lb, x)
        end

        if isinplace(prob)
            y = similar(prob.f.integrand_prototype)
            f = (x, dx) -> dx .= vec(prob.f(y, scale(x), p)) .* vol
        else
            y = prob.f(mid, p)
            f = (x, dx) -> dx .= Iterators.flatten(prob.f(scale(x), p)) .* vol
        end
        ncomp = length(y)
    end

    if alg isa CubaVegas
        out = Cuba.vegas(f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, seed = alg.seed, minevals = alg.minevals,
            nstart = alg.nstart, nincrease = alg.nincrease,
            gridno = alg.gridno)
    elseif alg isa CubaSUAVE
        out = Cuba.suave(f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, seed = alg.seed, minevals = alg.minevals,
            nnew = alg.nnew, nmin = alg.nmin, flatness = alg.flatness)
    elseif alg isa CubaDivonne
        out = Cuba.divonne(f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, seed = alg.seed, minevals = alg.minevals,
            key1 = alg.key1, key2 = alg.key2, key3 = alg.key3,
            maxpass = alg.maxpass, border = alg.border,
            maxchisq = alg.maxchisq, mindeviation = alg.mindeviation)
    elseif alg isa CubaCuhre
        out = Cuba.cuhre(f, ndim, ncomp; rtol = reltol,
            atol = abstol, nvec = nvec,
            maxevals = maxiters,
            flags = alg.flags, minevals = alg.minevals, key = alg.key)
    end

    # out.integral is a Vector{Float64}, but we want to return it to the shape of the integrand
    if prob.f isa BatchIntegralFunction
        if y isa AbstractVector
            val = out.integral[1]
        else
            val = reshape(out.integral, fsize)
        end
    else
        if y isa Real
            val = out.integral[1]
        elseif y isa AbstractVector
            val = out.integral
        else
            val = reshape(out.integral, size(y))
        end
    end

    SciMLBase.build_solution(prob, alg, val, out.error,
        chi = out.probability, retcode = ReturnCode.Success)
end

end