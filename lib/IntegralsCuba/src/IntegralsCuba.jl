module IntegralsCuba

using Integrals, Cuba
import Integrals: transformation_if_inf, scale_x, scale_x!

abstract type AbstractCubaAlgorithm <: SciMLBase.AbstractIntegralAlgorithm end
"""
    CubaVegas()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Importance sampling is used to reduce variance.

## References

@article{lepage1978new,
title={A new algorithm for adaptive multidimensional integration},
author={Lepage, G Peter},
journal={Journal of Computational Physics},
volume={27},
number={2},
pages={192--203},
year={1978},
publisher={Elsevier}
}
"""
struct CubaVegas <: AbstractCubaAlgorithm
    flags::Int
    seed::Int
    minevals::Int
    nstart::Int
    nincrease::Int
    gridno::Int
end
"""
    CubaSUAVE()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Suave stands for subregion-adaptive VEGAS.
Importance sampling and subdivision are thus used to reduce variance.

## References

@article{hahn2005cuba,
title={Cuba—a library for multidimensional numerical integration},
author={Hahn, Thomas},
journal={Computer Physics Communications},
volume={168},
number={2},
pages={78--95},
year={2005},
publisher={Elsevier}
}
"""
struct CubaSUAVE{R} <: AbstractCubaAlgorithm where {R <: Real}
    flags::Int
    seed::Int
    minevals::Int
    nnew::Int
    nmin::Int
    flatness::R
end
"""
    CubaDivonne()

Multidimensional adaptive Monte Carlo integration from Cuba.jl.
Stratified sampling is used to reduce variance.

## References

@article{friedman1981nested,
title={A nested partitioning procedure for numerical multiple integration},
author={Friedman, Jerome H and Wright, Margaret H},
journal={ACM Transactions on Mathematical Software (TOMS)},
volume={7},
number={1},
pages={76--92},
year={1981},
publisher={ACM New York, NY, USA}
}
"""
struct CubaDivonne{R1, R2, R3} <:
       AbstractCubaAlgorithm where {R1 <: Real, R2 <: Real, R3 <: Real}
    flags::Int
    seed::Int
    minevals::Int
    key1::Int
    key2::Int
    key3::Int
    maxpass::Int
    border::R1
    maxchisq::R2
    mindeviation::R3
end
"""
    CubaCuhre()

Multidimensional h-adaptive integration from Cuba.jl.

## References

@article{berntsen1991adaptive,
title={An adaptive algorithm for the approximate calculation of multiple integrals},
author={Berntsen, Jarle and Espelid, Terje O and Genz, Alan},
journal={ACM Transactions on Mathematical Software (TOMS)},
volume={17},
number={4},
pages={437--451},
year={1991},
publisher={ACM New York, NY, USA}
}
"""
struct CubaCuhre <: AbstractCubaAlgorithm
    flags::Int
    minevals::Int
    key::Int
end

function CubaVegas(; flags = 0, seed = 0, minevals = 0, nstart = 1000, nincrease = 500,
        gridno = 0)
    CubaVegas(flags, seed, minevals, nstart, nincrease, gridno)
end
function CubaSUAVE(; flags = 0, seed = 0, minevals = 0, nnew = 1000, nmin = 2,
        flatness = 25.0)
    CubaSUAVE(flags, seed, minevals, nnew, nmin, flatness)
end
function CubaDivonne(; flags = 0, seed = 0, minevals = 0,
        key1 = 47, key2 = 1, key3 = 1, maxpass = 5, border = 0.0,
        maxchisq = 10.0, mindeviation = 0.25)
    CubaDivonne(flags, seed, minevals, key1, key2, key3, maxpass, border, maxchisq,
        mindeviation)
end
CubaCuhre(; flags = 0, minevals = 0, key = 0) = CubaCuhre(flags, minevals, key)

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

export CubaVegas, CubaSUAVE, CubaDivonne, CubaCuhre

end
