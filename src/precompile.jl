@setup_workload begin
    f = (x, p) -> x^2
    prob = IntegralProblem(f, (0.0, 1.0))

    @compile_workload begin
        solve(prob, QuadGKJL(); reltol = 1.0e-8, abstol = 1.0e-8)
    end
end
