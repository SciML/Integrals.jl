using Pkg
using SafeTestsets
using Test

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path = subpkg_path))
end

dev_subpkg("IntegralsCuba")
dev_subpkg("IntegralsCubature")

@time @safetestset "Interface Tests" begin
    include("interface_tests.jl")
end
@time @safetestset "Derivative Tests" begin
    include("derivative_tests.jl")
end
@time @safetestset "Infinite Integral Tests" begin
    include("inf_integral_tests.jl")
end
@time @safetestset "Gaussian Quadrature Tests" begin
    include("gaussian_quadrature_tests.jl")
end
@time @safetestset "QuadratureFunction Tests" begin
    include("quadrule_tests.jl")
end
