function __solvebp_call(prob::SampledIntegralProblem, alg::TrapezoidalRule; kwargs...)
    dim = dimension(prob.dim)
    err = Inf64
    data = prob.y
    grid = prob.x
    # inlining is required in order to not allocate
    integrand = @inline function (i)
        # integrate along dimension `dim`, returning a n-1 dimensional array, or scalar if n=1
        _selectdim(data, dim, i)
    end

    firstidx, lastidx = firstindex(grid), lastindex(grid)

    out = integrand(firstidx)

    if isbits(out)
        # fast path for equidistant grids
        if grid isa AbstractRange
            dx = step(grid)
            out /= 2
            for i in (firstidx + 1):(lastidx - 1)
                out += integrand(i)
            end
            out += integrand(lastidx) / 2
            out *= dx
            # irregular grids:
        else
            out *= (grid[firstidx + 1] - grid[firstidx])
            for i in (firstidx + 1):(lastidx - 1)
                @inbounds out += integrand(i) * (grid[i + 1] - grid[i - 1])
            end
            out += integrand(lastidx) * (grid[lastidx] - grid[lastidx - 1])
            out /= 2
        end
    else # same, but inplace, broadcasted
        out = copy(out) # to prevent aliasing
        if grid isa AbstractRange
            dx = grid[begin + 1] - grid[begin]
            out ./= 2
            for i in (firstidx + 1):(lastidx - 1)
                out .+= integrand(i)
            end
            out .+= integrand(lastidx) ./ 2
            out .*= dx
        else
            out .*= (grid[firstidx + 1] - grid[firstidx])
            for i in (firstidx + 1):(lastidx - 1)
                @inbounds out .+= integrand(i) .* (grid[i + 1] - grid[i - 1])
            end
            out .+= integrand(lastidx) .* (grid[lastidx] - grid[lastidx - 1])
            out ./= 2
        end
    end
    return SciMLBase.build_solution(prob, alg, out, err, retcode = ReturnCode.Success)
end
