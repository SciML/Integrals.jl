using Integrals, Distributions, Test, Cubature, FastGaussQuadrature, StaticArrays

reltol = 0.0
abstol = 1e-3

# not all quadratures are compatible with infinities if they evaluate the endpoints
alg_req = Dict(
    QuadratureRule(gausslegendre, n = 50) => (
        nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
        allows_iip = false, allows_inf = true), QuadGKJL() => (
        nout = Inf, allows_batch = true, min_dim = 1, max_dim = 1,
        allows_iip = true, allows_inf = true),
    HCubatureJL() => (nout = Inf, allows_batch = false, min_dim = 1,
        max_dim = Inf, allows_iip = true, allows_inf = true), CubatureJLh() => (
        nout = Inf, allows_batch = true, min_dim = 1,
        max_dim = Inf, allows_iip = true, allows_inf = true)
)
# GaussLegendre(n=50) => (nout = Inf, min_dim = 1, max_dim = 1, allows_batch = false,
#     allows_iip = false, allows_inf=true),
# VEGAS() => (nout = 1, allows_batch = true, min_dim = 2, max_dim = Inf,
# allows_iip = true),
# CubatureJLp() => (nout = Inf, allows_batch = true, min_dim = 1,
#     max_dim = Inf, allows_iip = true, allows_inf=false),
# CubaVegas() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
#     allows_iip = true),
# CubaSUAVE() => (nout = Inf, allows_batch = true, min_dim = 1, max_dim = Inf,
#     allows_iip = true),
# CubaDivonne() => (nout = Inf, allows_batch = true, min_dim = 2,
#     max_dim = Inf, allows_iip = true),
# CubaCuhre() => (nout = Inf, allows_batch = true, min_dim = 2, max_dim = Inf,
#     allows_iip = true),
problems = (
    (; # 1. multi-variate infinite limits: Gaussian
        f = (x, p) -> pdf(MvNormal([0.00, 0.00], [0.4 0.0; 0.00 0.4]), x),
        domain = (@SVector[-Inf, -Inf], @SVector[Inf, Inf]),
        solution = 1.00
    ),
    (; # 2. multi-variate flipped infinite limits: Gaussian
        f = (x, p) -> pdf(MvNormal([0.00, 0.00], [0.4 0.0; 0.00 0.4]), x),
        domain = (@SVector[Inf, Inf], @SVector[-Inf, -Inf]),
        solution = 1.00
    ),
    (; # 3. multi-variate mixed infinite/semi-infinite upper limit: Gaussian
        f = (x, p) -> pdf(MvNormal([0.00, 0.00], [0.4 0.0; 0.00 0.4]), x),
        domain = (@SVector[-Inf, 0], @SVector[Inf, Inf]),
        solution = 0.5
    ),
    (; # 4. multi-variate mixed infinite/semi-infinite lower limit: Gaussian
        f = (x, p) -> pdf(MvNormal([0.00, 0.00], [0.4 0.0; 0.00 0.4]), x),
        domain = (@SVector[-Inf, -Inf], @SVector[Inf, 0]),
        solution = 0.5
    ),
    (; # 5. single-variable infinite limit: Gaussian
        f = (x, p) -> pdf(Normal(0.00, 1.00), x),
        domain = (-Inf, Inf),
        solution = 1.0
    ),
    (; # 6. single-variable flipped infinite limit: Gaussian
        f = (x, p) -> pdf(Normal(0.00, 1.00), x),
        domain = (Inf, -Inf),
        solution = -1.0
    ),
    (; # 7. single-variable semi-infinite upper limit: Gaussian
        f = (x, p) -> pdf(Normal(0.00, 1.00), x),
        domain = (0.00, Inf),
        solution = 0.5
    ),
    (; # 8. single-variable flipped, semi-infinite upper limit: Gaussian
        f = (x, p) -> pdf(Normal(0.00, 1.00), x),
        domain = (0.00, -Inf),
        solution = -0.5
    ),
    (; # 9. single-variable semi-infinite lower limit: Gaussian
        f = (x, p) -> pdf(Normal(0.00, 1.00), x),
        domain = (-Inf, 0.00),
        solution = 0.5
    ),
    (; # 10. single-variable flipped, semi-infinite lower limit: Gaussian
        f = (x, p) -> pdf(Normal(0.00, 1.00), x),
        domain = (Inf, 0.00),
        solution = -0.5
    ),
    (; # 11. single-variable infinite limit: Lorentzian
        f = (x, p) -> 1 / (x^2 + 1),
        domain = (-Inf, Inf),
        solution = pi / 1
    ),
    (; # 12. single-variable shifted, semi-infinite lower limit: Lorentzian
        f = (x, p) -> 1 / ((x - 2)^2 + 1),
        domain = (-Inf, 2),
        solution = pi / 2
    ),
    (; # 13. single-variable shifted, semi-infinite upper limit: Lorentzian
        f = (x, p) -> 1 / ((x - 2)^2 + 1),
        domain = (2, Inf),
        solution = pi / 2
    ),
    (; # 14. single-variable flipped, shifted, semi-infinite lower limit: Lorentzian
        f = (x, p) -> 1 / ((x - 2)^2 + 1),
        domain = (Inf, 2),
        solution = -pi / 2
    ),
    (; # 15. single-variable flipped, shifted, semi-infinite upper limit: Lorentzian
        f = (x, p) -> 1 / ((x - 2)^2 + 1),
        domain = (2, -Inf),
        solution = -pi / 2
    ),
    (; # 16. single-variable finite limits: constant
        f = (x, p) -> 1.0,
        domain = (1, 3),
        solution = 2
    ),
    (; # 17. single-variable flipped, finite limits: constant
        f = (x, p) -> 1.0,
        domain = (3, 1),
        solution = -2
    )
)

function f_helper!(f, y, x, p)
    y[] = f(x, p)
    return
end

function batch_helper(f, x, p)
    map(i -> f(x[axes(x)[begin:(end - 1)]..., i], p), axes(x)[end])
end

function batch_helper!(f, y, x, p)
    y .= batch_helper(f, x, p)
    return
end

do_tests = function (; f, domain, alg, abstol, reltol, solution)
    prob = IntegralProblem(f, domain)
    sol = solve(prob, alg; reltol, abstol, do_inf_transformation = Val(true))
    @test abs(only(sol) - solution) < max(abstol, reltol * abs(solution))
    cache = @test_nowarn @inferred init(prob, alg; do_inf_transformation = Val(true))
    @test_nowarn @inferred solve!(cache)
    @test_nowarn @inferred solve(prob, alg; do_inf_transformation = Val(true))
end

# IntegralFunction{false}
for (alg, req) in pairs(alg_req), (j, (; f, domain, solution)) in enumerate(problems)
    req.allows_inf || continue
    req.nout >= length(solution) || continue
    req.min_dim <= length(first(domain)) <= req.max_dim || continue

    @info "oop infinity test" alg=nameof(typeof(alg)) problem=j
    do_tests(; f, domain, solution, alg, abstol, reltol)
end

# IntegralFunction{true}
for (alg, req) in pairs(alg_req), (j, (; f, domain, solution)) in enumerate(problems)
    req.allows_inf || continue
    req.nout >= length(solution) || continue
    req.allows_iip || continue
    req.min_dim <= length(first(domain)) <= req.max_dim || continue

    @info "iip infinity test" alg=nameof(typeof(alg)) problem=j
    fiip = IntegralFunction((y, x, p) -> f_helper!(f, y, x, p), zeros(size(solution)))
    do_tests(; f = fiip, domain, solution, alg, abstol, reltol)
end

# BatchIntegralFunction{false}
for (alg, req) in pairs(alg_req), (j, (; f, domain, solution)) in enumerate(problems)
    req.allows_inf || continue
    req.nout >= length(solution) || continue
    req.allows_batch || continue
    req.min_dim <= length(first(domain)) <= req.max_dim || continue

    @info "Batched, oop infinity test" alg=nameof(typeof(alg)) problem=j
    bf = BatchIntegralFunction((x, p) -> batch_helper(f, x, p))
    do_tests(; f = bf, domain, solution, alg, abstol, reltol)
end

# BatchIntegralFunction{true}
for (alg, req) in pairs(alg_req), (j, (; f, domain, solution)) in enumerate(problems)
    req.allows_inf || continue
    req.nout >= length(solution) || continue
    req.allows_batch || continue
    req.allows_iip || continue
    req.min_dim <= length(first(domain)) <= req.max_dim || continue

    @info "Batched, iip infinity test" alg=nameof(typeof(alg)) problem=j
    bfiip = BatchIntegralFunction((y, x, p) -> batch_helper!(f, y, x, p), zeros(0))
    do_tests(; f = bfiip, domain, solution, alg, abstol, reltol)
end
