using ExplicitImports
using Integrals
using Test
using LinearAlgebra: norm

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(Integrals) === nothing
    # Note: norm is used in default parameters in included files (algorithms.jl)
    # which ExplicitImports may not detect properly, so we ignore it
    @test check_no_stale_explicit_imports(Integrals; ignore = (:norm, :AbstractVerbositySpecifier, 
        :DebugLevel, :Detailed, :ErrorLevel, :Minimal, :None, :Standard, :All, :AbstractMessageLevel)) === nothing
end
