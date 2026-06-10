using Pkg
using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")

# QA (Aqua + ExplicitImports + JET) runs in an isolated environment (test/qa) so
# its tooling deps never enter the main test target's resolve. On Julia < 1.11
# the [sources] table is ignored, so develop the package by path to test the PR
# branch code.
function activate_qa_env()
    Pkg.activate(joinpath(@__DIR__, "qa"))
    if VERSION < v"1.11.0-DEV.0"
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    end
    return Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
    @time @safetestset "Interface Tests" include("interface_tests.jl")
    @time @safetestset "Infinite Integral Tests" include("inf_integral_tests.jl")
    @time @safetestset "Gaussian Quadrature Tests" include("gaussian_quadrature_tests.jl")
    @time @safetestset "Sampled Integration Tests" include("sampled_tests.jl")
    @time @safetestset "QuadratureFunction Tests" include("quadrule_tests.jl")
    @time @safetestset "Verbosity Tests" include("verbosity_tests.jl")
    @time @safetestset "Alternative Transformation Tests" include("alt_transformation_tests.jl")
    @time @safetestset "HAdaptiveIntegration Tests" include("hadaptiveintegration_tests.jl")
end

if GROUP == "All" || GROUP == "AD"
    @time @safetestset "Derivative Tests" include("derivative_tests.jl")
    @time @safetestset "Nested AD Tests" include("nested_ad_tests.jl")
end

if GROUP == "QA"
    activate_qa_env()
    @time @safetestset "Quality Assurance" include("qa/qa.jl")
end
