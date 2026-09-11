using Integrals, BenchmarkTools

const SUITE = BenchmarkGroup()

# =============================================================================
# Scalar (1D) integrals
# =============================================================================

SUITE["scalar"] = BenchmarkGroup()

f_sin(x, p) = sin(x * p)
prob_sin = IntegralProblem(f_sin, (0.0, π), 2.0)

f_smooth(x, p) = exp(-p * x^2)
prob_gauss = IntegralProblem(f_smooth, (-5.0, 5.0), 1.0)

SUITE["scalar"]["QuadGKJL"] = @benchmarkable solve(
    $prob_sin, QuadGKJL(); reltol = 1.0e-8, abstol = 1.0e-8
)
SUITE["scalar"]["QuadGKJL_gauss"] = @benchmarkable solve(
    $prob_gauss, QuadGKJL(); reltol = 1.0e-8, abstol = 1.0e-8
)
SUITE["scalar"]["HCubatureJL"] = @benchmarkable solve(
    $prob_sin, HCubatureJL(); reltol = 1.0e-4, abstol = 1.0e-4
)

# =============================================================================
# Multidimensional integrals
# =============================================================================

SUITE["multidim"] = BenchmarkGroup()

f_2d(x, p) = sin(x[1]) * cos(x[2]) * p
prob_2d = IntegralProblem(f_2d, ([0.0, 0.0], [π, π]), 1.0)

f_3d(x, p) = exp(-(x[1]^2 + x[2]^2 + x[3]^2) / p)
prob_3d = IntegralProblem(f_3d, ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]), 1.0)

SUITE["multidim"]["HCubatureJL_2d"] = @benchmarkable solve(
    $prob_2d, HCubatureJL(); reltol = 1.0e-4, abstol = 1.0e-4
)
SUITE["multidim"]["HCubatureJL_3d"] = @benchmarkable solve(
    $prob_3d, HCubatureJL(); reltol = 1.0e-4, abstol = 1.0e-4
)
SUITE["multidim"]["VEGAS_2d"] = @benchmarkable solve(
    $prob_2d, VEGAS(); reltol = 1.0e-2, abstol = 1.0e-2
)

# =============================================================================
# In-place (out-of-place vs in-place interface)
# =============================================================================

SUITE["inplace"] = BenchmarkGroup()

function f_ip(dx, x, p)
    dx[1] = sin(x) * p
    return nothing
end
prob_ip = IntegralProblem(IntegralFunction(f_ip, zeros(1)), (0.0, π), 2.0)

SUITE["inplace"]["QuadGKJL"] = @benchmarkable solve(
    $prob_ip, QuadGKJL(); reltol = 1.0e-8, abstol = 1.0e-8
)
