using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @time @safetestset "Explicit Imports" include("explicit_imports_tests.jl")
    @time @safetestset "Quality Assurance" include("qa.jl")
    @time @safetestset "Interface Tests" include("interface_tests.jl")
    @time @safetestset "Infinite Integral Tests" include("inf_integral_tests.jl")
    @time @safetestset "Gaussian Quadrature Tests" include("gaussian_quadrature_tests.jl")
    @time @safetestset "Sampled Integration Tests" include("sampled_tests.jl")
    @time @safetestset "QuadratureFunction Tests" include("quadrule_tests.jl")
    @time @safetestset "Alternative Transformation Tests" include("alt_transformation_tests.jl")
end

if GROUP == "All" || GROUP == "AD"
    @time @safetestset "Derivative Tests" include("derivative_tests.jl")
    @time @safetestset "Nested AD Tests" include("nested_ad_tests.jl")
end
