@setup_workload begin
    f = (x, p) -> x^2
    prob = IntegralProblem(f, (0.0, 1.0))
    f! = (y, x, p) -> (y[1] = x^2; nothing)
    probi = IntegralProblem(IntegralFunction(f!, zeros(1)), (0.0, 1.0))
    g = (x, p) -> x[1] * x[2]
    prob2 = IntegralProblem(g, ([0.0, 0.0], [1.0, 1.0]))

    @compile_workload begin
        solve(prob, QuadGKJL(); reltol = 1.0e-8, abstol = 1.0e-8)
        solve(probi, QuadGKJL())
        solve(prob, HCubatureJL())
        solve(prob2, HCubatureJL())
    end
end
