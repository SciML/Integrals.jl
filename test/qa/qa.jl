using SciMLTesting, Integrals, Test
using JET

run_qa(
    Integrals;
    explicit_imports = true,
    aqua_kwargs = (;
        # IntegralProblem / SampledIntegralProblem are SciMLBase types this package
        # owns the integral-solver methods for, so dispatching on them is not piracy.
        piracies = (; treat_as_own = [IntegralProblem, SampledIntegralProblem]),
    ),
    ei_kwargs = (;
        no_stale_explicit_imports = (;
            # Referenced only inside `@verbosity_specifier IntegralVerbosity` (src/verbosity.jl):
            # the macro generates `IntegralVerbosity(::None)` / `(::Minimal)` / ... preset
            # constructors and `MessageLevel` / `AbstractVerbositySpecifier` /
            # `AbstractVerbosityPreset` type guards that need these bare names in `Integrals`'s
            # scope. ExplicitImports cannot see through the macro, so it reports them stale;
            # dropping the imports breaks the constructor at runtime
            # (`IntegralVerbosity(; preset=Standard())` -> UndefVarError: `AbstractVerbosityPreset`).
            ignore = (
                :AbstractVerbositySpecifier, :AbstractVerbosityPreset, :MessageLevel,
                :None, :Minimal, :Standard, :Detailed, :All,
            ),
        ),
    ),
)

# Type-stability (JET opt-mode) regression guards for the hot solver paths. These are
# repo-specific @report_opt checks, orthogonal to run_qa's package-level JET typo check.
@testset "JET opt-mode solver paths" begin
    @testset "QuadGKJL" begin
        f = (x, p) -> x^2
        prob = IntegralProblem(f, (0.0, 1.0))
        rep = @report_opt target_modules = (Integrals,) solve(prob, QuadGKJL())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "HCubatureJL" begin
        f = (x, p) -> x[1]^2 + x[2]^2
        prob = IntegralProblem(f, ([0.0, 0.0], [1.0, 1.0]))
        rep = @report_opt target_modules = (Integrals,) solve(prob, HCubatureJL())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "SampledIntegralProblem with TrapezoidalRule" begin
        x = range(0, 1, length = 100)
        y = x .^ 2
        prob = SampledIntegralProblem(y, x)
        rep = @report_opt target_modules = (Integrals,) solve(prob, TrapezoidalRule())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "SampledIntegralProblem with SimpsonsRule" begin
        x = range(0, 1, length = 101)
        y = x .^ 2
        prob = SampledIntegralProblem(y, x)
        rep = @report_opt target_modules = (Integrals,) solve(prob, SimpsonsRule())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "Infinite bounds transformation" begin
        f = (x, p) -> exp(-x^2)
        prob = IntegralProblem(f, (0.0, Inf))
        rep = @report_opt target_modules = (Integrals,) solve(prob, QuadGKJL())
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "VEGAS" begin
        # VEGAS has some inherent type instability issues due to:
        # 1. Captured variables in closures (necessary for in-place operations)
        # 2. Runtime dispatch for integrand type checking
        # We verify the number of issues is bounded and doesn't regress.
        f = (x, p) -> x^2
        prob = IntegralProblem(f, (0.0, 1.0))
        rep = @report_opt target_modules = (Integrals,) solve(prob, VEGAS())
        @test length(JET.get_reports(rep)) <= 2
    end
end
