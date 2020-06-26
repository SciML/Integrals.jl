using Pkg
using SafeTestsets
using Test

@time @safetestset "Interface Tests" begin include("interface_tests.jl") end
@time @safetestset "Derivative Tests" begin include("derivative_tests.jl") end
