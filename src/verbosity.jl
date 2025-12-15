@verbosity_specifier IntegralVerbosity begin
    toggles = (
        :cache_init,
        :domain_transformation,
        :algorithm_selection,
        :iteration_progress,
        :convergence_result,
        :batch_mode,
        :buffer_allocation,
        :deprecations
    )

    presets = (
        Standard = (
            cache_init = InfoLevel(),
            domain_transformation = InfoLevel(),
            algorithm_selection = DebugLevel(),
            iteration_progress = Silent(),
            convergence_result = InfoLevel(),
            batch_mode = DebugLevel(),
            buffer_allocation = DebugLevel(),
            deprecations = WarnLevel()
        ),
    )

    groups = (
        solver = (:iteration_progress, :convergence_result),
        setup = (:cache_init, :domain_transformation, :algorithm_selection),
        debug = (:batch_mode, :buffer_allocation)
    )
end

"""
    IntegralVerbosity

Verbosity specifier for controlling logging output in Integrals.jl.

# Toggles

- `:cache_init` - IntegralCache initialization and setup
- `:domain_transformation` - Change of variables for infinite bounds and domain remapping
- `:algorithm_selection` - Which algorithm is being used and its parameters
- `:iteration_progress` - Per-iteration updates during adaptive integration (can be verbose)
- `:convergence_result` - Final integration result with value, error estimate, and tolerances
- `:batch_mode` - Batch evaluation mode selection (in-place vs out-of-place)
- `:buffer_allocation` - Pre-allocated buffer creation for reusable caches
- `:deprecations` - Deprecation warnings for outdated API usage

# Usage

## Using Presets

```julia
# Standard preset (default)
solve(prob, QuadGKJL(); verbose = IntegralVerbosity())

# Completely silent
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(None()))
```

## Setting Individual Toggles

```julia
# Show only convergence results
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(
    cache_init = Silent(),
    domain_transformation = Silent(),
    algorithm_selection = Silent(),
    iteration_progress = Silent(),
    convergence_result = InfoLevel(),
    batch_mode = Silent(),
    buffer_allocation = Silent()
))

# Enable iteration-by-iteration progress (useful for slow integrals)
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(iteration_progress = InfoLevel()))
```

## Using Groups

```julia
# Enable all solver messages (iteration progress + convergence)
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(solver = InfoLevel()))

# Enable all setup messages (cache, domain transforms, algorithm selection)
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(setup = InfoLevel()))

# Enable debug information (batch mode + buffer allocation)
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(debug = DebugLevel()))
```

# Default Levels (Standard Preset)

- `cache_init` → `InfoLevel()` (visible by default)
- `domain_transformation` → `InfoLevel()` (visible by default)
- `algorithm_selection` → `DebugLevel()` (requires `ENV["JULIA_DEBUG"] = "Integrals"`)
- `iteration_progress` → `Silent()` (off by default, can be very verbose)
- `convergence_result` → `InfoLevel()` (visible by default)
- `batch_mode` → `DebugLevel()` (requires `ENV["JULIA_DEBUG"] = "Integrals"`)
- `buffer_allocation` → `DebugLevel()` (requires `ENV["JULIA_DEBUG"] = "Integrals"`)
- `deprecations` → `WarnLevel()` (visible by default as warnings)
"""
function IntegralVerbosity end 

const DEFAULT_VERBOSE = ODEVerbosity()

@inline function _process_verbose_param(verbose::SciMLLogging.AbstractVerbosityPreset)
    IntegralVerbosity(verbose)
end

@inline function _process_verbose_param(verbose::Bool)
    verbose ? DEFAULT_VERBOSE : IntegralVerbosity(SciMLLogging.None())
end

@inline _process_verbose_param(verbose::IntegralVerbosity) = verbose
