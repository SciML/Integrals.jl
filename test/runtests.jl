if get(ENV, "GROUP", "All") == "QA"
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.instantiate()
end

using SciMLTesting
run_tests()
