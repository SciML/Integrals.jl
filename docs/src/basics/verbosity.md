# Verbosity Control

## Verbosity Specification with SciMLLogging.jl

Integrals.jl uses SciMLLogging.jl to provide users with fine-grained control over logging and diagnostic output during integration. The `IntegralVerbosity` struct allows you to customize which messages are displayed, from deprecation warnings to detailed debugging information about algorithm selection, convergence, and iteration progress.

## Basic Usage

Pass an `IntegralVerbosity` object to `solve` or `init` using the `verbose` keyword argument:

```julia
using Integrals
using Integrals: IntegralVerbosity

# Define an integral problem
f(x, p) = x^2  # p is unused but required by the IntegralProblem interface
prob = IntegralProblem(f, 0.0, 1.0)

# Solve with detailed verbosity to see algorithm info and convergence
verbose = IntegralVerbosity(Detailed())
sol = solve(prob, QuadGKJL(), verbose = verbose)

# Solve with completely silent output (no deprecation warnings)
sol = solve(prob, QuadGKJL(), verbose = IntegralVerbosity(None()))

# Solve with default verbosity (only deprecation warnings)
sol = solve(prob, QuadGKJL())  # equivalent to verbose = IntegralVerbosity()
```

## API Reference

```@docs
Integrals.IntegralVerbosity
```
