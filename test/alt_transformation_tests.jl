using Integrals, Test

reltol = 1e-6
abstol = 1e-6

@testset "Alternative Infinity Transformations" begin
    @testset "transformation_tan_inf" begin
        # Test doubly-infinite integral: ∫_{-∞}^{∞} exp(-x²) dx = √π
        f_gaussian = (x, p) -> exp(-x^2)
        prob = IntegralProblem(f_gaussian, (-Inf, Inf))
        alg = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, sqrt(π), rtol = 1e-5)

        # Test Lorentzian: ∫_{-∞}^{∞} 1/(1+x²) dx = π
        f_lorentz = (x, p) -> 1 / (1 + x^2)
        prob = IntegralProblem(f_lorentz, (-Inf, Inf))
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, π, rtol = 1e-5)

        # Test semi-infinite upper: ∫_0^{∞} exp(-x) dx = 1
        f_exp = (x, p) -> exp(-x)
        prob = IntegralProblem(f_exp, (0.0, Inf))
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, 1.0, rtol = 1e-4)

        # Test semi-infinite lower: ∫_{-∞}^0 exp(x) dx = 1
        f_exp_neg = (x, p) -> exp(x)
        prob = IntegralProblem(f_exp_neg, (-Inf, 0.0))
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, 1.0, rtol = 1e-4)

        # Test semi-infinite with non-zero bound: ∫_2^{∞} 1/((x-2)² + 1) dx = π/2
        f_shifted_lorentz = (x, p) -> 1 / ((x - 2)^2 + 1)
        prob = IntegralProblem(f_shifted_lorentz, (2.0, Inf))
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, π / 2, rtol = 1e-4)

        # Test with negative semi-infinite bound: ∫_{-∞}^{-1} exp(x) dx = exp(-1)
        f_exp_shifted = (x, p) -> exp(x)
        prob = IntegralProblem(f_exp_shifted, (-Inf, -1.0))
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, exp(-1), rtol = 1e-4)
    end

    @testset "transformation_cot_inf" begin
        # Test doubly-infinite integral: ∫_{-∞}^{∞} exp(-x²) dx = √π
        f_gaussian = (x, p) -> exp(-x^2)
        prob = IntegralProblem(f_gaussian, (-Inf, Inf))
        alg = ChangeOfVariables(transformation_cot_inf, QuadGKJL())
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, sqrt(π), rtol = 1e-5)

        # Test semi-infinite upper: ∫_0^{∞} exp(-x) dx = 1
        f_exp = (x, p) -> exp(-x)
        prob = IntegralProblem(f_exp, (0.0, Inf))
        sol = solve(prob, alg; reltol = 1e-4, abstol = 1e-4)
        @test isapprox(sol.u, 1.0, rtol = 1e-3)

        # Test semi-infinite lower: ∫_{-∞}^0 exp(x) dx = 1
        f_exp_neg = (x, p) -> exp(x)
        prob = IntegralProblem(f_exp_neg, (-Inf, 0.0))
        sol = solve(prob, alg; reltol = 1e-4, abstol = 1e-4)
        @test isapprox(sol.u, 1.0, rtol = 1e-3)

        # Test half Gaussian: ∫_0^{∞} exp(-x²) dx = √π/2
        prob = IntegralProblem(f_gaussian, (0.0, Inf))
        sol = solve(prob, alg; reltol = 1e-4, abstol = 1e-4)
        @test isapprox(sol.u, sqrt(π) / 2, rtol = 1e-3)
    end

    @testset "Finite domains unchanged" begin
        # Finite domains should work the same with all transformations
        f = (x, p) -> x^2
        prob = IntegralProblem(f, (0.0, 1.0))

        sol_default = solve(prob, QuadGKJL(); reltol, abstol)
        sol_tan = solve(prob, ChangeOfVariables(transformation_tan_inf, QuadGKJL());
            reltol, abstol)
        sol_cot = solve(prob, ChangeOfVariables(transformation_cot_inf, QuadGKJL());
            reltol, abstol)

        @test isapprox(sol_default.u, 1 / 3, rtol = 1e-8)
        @test isapprox(sol_tan.u, 1 / 3, rtol = 1e-8)
        @test isapprox(sol_cot.u, 1 / 3, rtol = 1e-8)
    end

    @testset "Flipped infinite limits" begin
        # Test with flipped bounds (negative orientation)
        f = (x, p) -> exp(-x^2)

        # ∫_{∞}^{-∞} exp(-x²) dx = -√π
        prob = IntegralProblem(f, (Inf, -Inf))
        alg = ChangeOfVariables(transformation_tan_inf, QuadGKJL())
        sol = solve(prob, alg; reltol, abstol)
        @test isapprox(sol.u, -sqrt(π), rtol = 1e-5)
    end

    @testset "Works with different algorithms" begin
        f = (x, p) -> exp(-x^2)
        prob = IntegralProblem(f, (-Inf, Inf))

        # Test with HCubatureJL
        alg_hcub = ChangeOfVariables(transformation_tan_inf, HCubatureJL())
        sol = solve(prob, alg_hcub; reltol, abstol)
        @test isapprox(sol.u, sqrt(π), rtol = 1e-4)
    end

    @testset "ChangeOfVariables exported" begin
        # Verify ChangeOfVariables is exported and accessible
        @test ChangeOfVariables isa UnionAll  # Has type parameters
        @test transformation_if_inf isa Function
        @test transformation_tan_inf isa Function
        @test transformation_cot_inf isa Function
    end
end
