using SciMLTesting, Integrals, Test
using JET

# ExplicitImports only checks an extension module once it actually exists, and an
# extension is only loaded once every one of its triggers is. Loading all of the
# weakdeps here is what makes `run_qa` scan ext/ at all.
using ADTypes, Arblib, ChainRulesCore, Cuba, Cubature, DifferentiationInterface
using FastGaussQuadrature, FastTanhSinhQuadrature, ForwardDiff, HAdaptiveIntegration
using MCIntegration, Mooncake, Zygote, ZygoteRules

run_qa(
    Integrals;
    reexports_allow = (
        :BatchIntegralFunction, :IntegralFunction, :IntegralProblem, :ReturnCode,
        :SampledIntegralProblem, :SciMLBase, :init, :isinplace, :remake, :solve, :solve!,
    ),
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
        all_qualified_accesses_are_public = (;
            ignore = (
                # Integrals' own internals. The `ext/` modules are part of this package,
                # so they implement and call its unexported solver interface directly;
                # there is no public spelling of any of these and exporting them would
                # commit the package to a SemVer-stable internal solver API.
                :DEFAULT_VERBOSE, :IntegralCache, :AbstractIntegralCExtensionAlgorithm,
                :MooncakeVJP, :ZygoteVJP, :ReverseDiffVJP,
                :__solvebp, :__solvebp_call, :_compute_dfdp_and_f, :_evaluate!,
                :build_problem, :checkkwargs, :gausslegendre, :get_prototype,
                :init_cacheval, :substitute_bv, :substitute_f, :substitute_v,
                :t2ujac, :u2t,
                # Arblib: the ball-integration entry points and the `Acb` setter are
                # unexported and Arblib declares nothing `public`.
                :integrate, :integrate!, :set!,
                # ForwardDiff: the dual-number type and its element-type guards are the
                # documented AD interface but are not exported or declared public.
                :Dual, :can_dual, :throw_cannot_dual,
                # QuadGK: `cachedrule` is the rule cache the Mooncake rules must mark
                # non-differentiable; not public.
                :cachedrule,
                # Zygote: `Buffer` is the documented way to write mutating code under
                # Zygote, but it is not exported or declared public.
                :Buffer,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                # Integrals' own internals, imported by its own `ext/` modules (see above).
                :AbstractCubaAlgorithm, :AbstractCubatureJLAlgorithm,
                :AbstractIntegralMetaAlgorithm, :scale_x, :scale_x!,
                # Mooncake's rule-definition interface: the only way to register rules
                # with Mooncake, and none of it is exported or declared public.
                Symbol("@from_chainrules"), Symbol("@zero_derivative"), :MinimalCtx,
                # ZygoteRules owns `literal_getproperty` (Zygote only re-exports it) and
                # does not declare it public; overloading it is the only way to give
                # `sol.u` an adjoint.
                :literal_getproperty,
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
        # JET's opt-mode analyzer misinterprets 1.13 optimized IR: it reports runtime
        # dispatch on fully concrete code here (and crashes outright on JET 0.12+,
        # https://github.com/aviatesk/JET.jl/issues/863). Native `Base.return_types`
        # on this path is concrete on 1.12 and 1.13.
        @test length(JET.get_reports(rep)) == 0 broken = VERSION >= v"1.13"
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
        # JET's opt-mode analyzer misinterprets 1.13 optimized IR: it reports runtime
        # dispatch on fully concrete code here (and crashes outright on JET 0.12+,
        # https://github.com/aviatesk/JET.jl/issues/863). Native `Base.return_types`
        # on this path is concrete on 1.12 and 1.13.
        @test length(JET.get_reports(rep)) <= 2 broken = VERSION >= v"1.13"
    end
end
