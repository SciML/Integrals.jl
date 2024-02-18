# Frequently Asked Questions

## How should I use the in-place interface?

The in-place interface allows evaluating vector-valued integrands without
allocating an output array. This can be beneficial for reducing allocations when
integrating many functions simultaneously or to make use of existing in-place
code. However, note that not all algorithms use in-place operations under the
hood, i.e. [`HCubatureJL`](@ref), and may still allocate.

You can construct an `IntegralFunction(f, prototype)`, where `f` is of the form
`f(y, u, p)` where `prototype` is of the desired type and shape of `y`.

For small array outputs of a known size, consider using StaticArrays.jl for the
return value of your integrand.

## How should I use the batch interface?

The batch interface allows evaluating one (or more) integrals simultaneously at
different points, which maximizes the parallelism for a given algorithm.

You can construct an out-of-place `BatchIntegralFunction(bf)` where `bf` is of
the form `bf(u, p) = stack(x -> f(x, p), eachslice(u; dims=ndims(u)))`, where
`f` is the (unbatched) integrand.
For interoperability with as many algorithms as possible, it is important that your out-of-place batch integrand accept an **empty** array of quadrature points and still return an output with a size and type consistent with the non-empty case.

You can construct an in-place `BatchIntegralFunction(bf, prototype)`, where `bf`
is of the form `bf(y, u, p) = foreach((y,x) -> f(y,x,p), eachslice(y, dims=ndims(y)), eachslice(x, dims=ndims(x)))`.

Note that not all algorithms use in-place batched operations under the hood,
i.e. [`QuadGKJL`](@ref).

## What should I do if my solution is not converged?

Certain algorithms, such as [`QuadratureRule`](@ref) used a fixed number of points to
calculate an integral and cannot provide an error estimate. In this case, you
have to increase the number of points and check the convergence yourself, which
will depend on the accuracy of the rule you choose.

For badly-behaved integrands, such as (nearly) singular and highly oscillatory
functions, most algorithms will fail to converge and either throw an error or
silently return the incorrect result. In some cases Integrals.jl can provide an
error code when things go wrong, but otherwise you can always check if the error
estimate for the integral is less than the requested tolerance, e.g. `sol.resid
< max(abstol, reltol*norm(sol.u))`. Sometimes using a larger tolerance or higher
precision arithmetic may help.

## How can I integrate arbitrarily-spaced data?

See [`SampledIntegralProblem`](@ref).

## How can I integrate on arbitrary geometries?

You can't, since Integrals.jl currently supports integration on hypercubes
because that is what lower-level packages implement.

## I don't see algorithm X or quadrature scheme Y ?

Fixed quadrature rules from other packages can be used with `QuadratureRule`.
Otherwise, feel free to open an issue or pull request.

## My integrand works with algorithm X but fails on algorithm Y

While bugs are not out of the question, certain algorithms, especially those implemented in C, are not compatible with arbitrary Julia types and have to return specific numeric types or arrays thereof.
In some cases, such as [`ArblibJL`](@ref), it is also expected that the integrand work with a custom quadrature point type.
Moreover, some algorithms, such as [`VEGAS`](@ref), only support scalar integrands.
For more details see the [solver page](solvers).

## Can I take derivatives with respect to the limits of integration?

Currently this is not implemented.