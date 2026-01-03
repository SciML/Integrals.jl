using SafeTestsets
using Test

@time @safetestset "Explicit Imports" include("explicit_imports_tests.jl")
@time @safetestset "Quality Assurance" include("qa.jl")
@time @safetestset "Interface Tests" include("interface_tests.jl")
# Skip AD tests on pre-release Julia due to potential AD package incompatibilities
if VERSION.prerelease == ()
    @time @safetestset "Derivative Tests" include("derivative_tests.jl")
else
    @warn "Skipping Derivative Tests on pre-release Julia $(VERSION)"
end
@time @safetestset "Infinite Integral Tests" include("inf_integral_tests.jl")
@time @safetestset "Gaussian Quadrature Tests" include("gaussian_quadrature_tests.jl")
@time @safetestset "Sampled Integration Tests" include("sampled_tests.jl")
@time @safetestset "QuadratureFunction Tests" include("quadrule_tests.jl")
# Skip Nested AD tests on pre-release Julia due to potential AD package incompatibilities
if VERSION.prerelease == ()
    @time @safetestset "Nested AD Tests" include("nested_ad_tests.jl")
else
    @warn "Skipping Nested AD Tests on pre-release Julia $(VERSION)"
end
