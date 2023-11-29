using Integrals, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(Integrals)
    Aqua.test_ambiguities(Integrals, recursive = false)
    Aqua.test_deps_compat(Integrals)
    Aqua.test_piracies(Integrals,
        treat_as_own = [IntegralProblem, SampledIntegralProblem])
    Aqua.test_project_extras(Integrals)
    Aqua.test_stale_deps(Integrals)
    Aqua.test_unbound_args(Integrals)
    Aqua.test_undefined_exports(Integrals)
end
