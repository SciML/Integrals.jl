# Integrals with Caching Interface

Often, integral solvers allocate memory or reuse quadrature rules for solving different
problems. For example, if one is going to solve the same integral for several parameters

```julia
using Integrals

prob = IntegralProblem((x, p) -> sin(x * p), 0, 1, 14.0)
alg = QuadGKJL()

solve(prob, alg)

prob = remake(prob, p = 15.0)
solve(prob, alg)
```

then it would be more efficient to allocate the heap used by `quadgk` across several calls,
shown below by directly calling the library

```julia
using QuadGK
segbuf = QuadGK.alloc_segbuf()
quadgk(x -> sin(14x), 0, 1, segbuf = segbuf)
quadgk(x -> sin(15x), 0, 1, segbuf = segbuf)
```

Integrals.jl's caching interface automates this process to reuse resources if an algorithm
supports it and if the necessary types to build the cache can be inferred from `prob`. To do
this with Integrals.jl, you simply `init` a cache, `solve!`, replace `p`, and solve again.
This uses the [SciML `init` interface](https://docs.sciml.ai/SciMLBase/stable/interfaces/Init_Solve/#init-and-the-Iterator-Interface)

```@example cache1
using Integrals

prob = IntegralProblem((x, p) -> sin(x * p), 0, 1, 14.0)
alg = QuadGKJL()

cache = init(prob, alg)
sol1 = solve!(cache)
```

```@example cache1
cache.p = 15.0
sol2 = solve!(cache)
```

The caching interface is intended for updating `p`, `lb`, `ub`, `nout`, and `batch`.
Note that the types of these variables is not allowed to change.
If it is necessary to change the integrand `f` instead of defining a new
`IntegralProblem`, consider using
[FunctionWrappers.jl](https://github.com/yuyichao/FunctionWrappers.jl).
