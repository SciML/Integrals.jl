# Integrals with Caching Interface

Often, integral solvers allocate memory or reuse quadrature rules for solving different
problems. For example, if one is going to perform
```julia
using Integrals

prob = IntegralProblem((x,p) -> sin(x*p), 0, 1, 14.0)
alg = QuadGKJL()

solve(prob, alg)

prob = remake(prob, f=(x,p) -> cos(x*p))
solve(prob, alg)
```
then it would be more efficient to allocate the heap used by `quadgk` across several calls,
shown below by directly calling the library
```julia
using QuadGK
segbuf = QuadGK.alloc_segbuf()
quadgk(x -> sin(15x), 0, 1, segbuf=segbuf)
quadgk(x -> cos(15x), 0, 1, segbuf=segbuf)
```
Integrals.jl's caching interface automates this process to reuse resources if an algorithm
supports it and if the necessary types to build the cache can be inferred from `prob`. To do
this with Integrals.jl, you simply `init` a cache, `solve`, replace `f`, and solve again.
This looks like
```@example cache1
using Integrals

prob = IntegralProblem((x,p) -> sin(x*p), 0, 1, 14.0)
alg = QuadGKJL()

cache = init(prob, alg)
sol1 = solve!(cache)
```

```@example cache1
cache = Integrals.set_f(cache, (x,p) -> cos(x*p))
sol2 = solve!(cache)
```
Similar cache-rebuilding functions are provided, including: `set_p`, `set_lb`, and `set_ub`,
each of which provides a new value of `lb`, `ub`, or `p`, respectively. When resetting the
cache, new allocations may be needed if those inferred types change.