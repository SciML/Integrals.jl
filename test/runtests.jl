using Pkg
using SafeTestsets
using Test

@time @safetestset "Interface Tests" include("interface_tests.jl")
@time @safetestset "Derivative Tests" include("derivative_tests.jl")
@time @safetestset "Infinite Integral Tests" include("inf_integral_tests.jl")
@time @safetestset "Gaussian Quadrature Tests" include("gaussian_quadrature_tests.jl")
@time @safetestset "Sampled Integration Tests" include("sampled_tests.jl")
@time @safetestset "QuadratureFunction Tests" include("quadrule_tests.jl")
@time @safetestset "Nested AD Tests" include("nested_ad_tests.jl")