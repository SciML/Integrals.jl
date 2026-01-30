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
        None = (
            cache_init = Silent(),
            domain_transformation = Silent(),
            algorithm_selection = Silent(),
            iteration_progress = Silent(),
            convergence_result = Silent(),
            batch_mode = Silent(),
            buffer_allocation = Silent(),
            deprecations = Silent()
        ),
        Minimal = (
            cache_init = Silent(),
            domain_transformation = Silent(),
            algorithm_selection = Silent(),
            iteration_progress = Silent(),
            convergence_result = Silent(),
            batch_mode = Silent(),
            buffer_allocation = Silent(),
            deprecations = WarnLevel()
        ),
        Standard = (
            cache_init = Silent(),
            domain_transformation = Silent(),
            algorithm_selection = Silent(),
            iteration_progress = Silent(),
            convergence_result = Silent(),
            batch_mode = Silent(),
            buffer_allocation = Silent(),
            deprecations = WarnLevel()
        ),
        Detailed = (
            cache_init = InfoLevel(),
            domain_transformation = InfoLevel(),
            algorithm_selection = InfoLevel(),
            iteration_progress = Silent(),
            convergence_result = InfoLevel(),
            batch_mode = InfoLevel(),
            buffer_allocation = InfoLevel(),
            deprecations = WarnLevel()
        ),
        All = (
            cache_init = InfoLevel(),
            domain_transformation = InfoLevel(),
            algorithm_selection = InfoLevel(),
            iteration_progress = InfoLevel(),
            convergence_result = InfoLevel(),
            batch_mode = InfoLevel(),
            buffer_allocation = InfoLevel(),
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

# Presets

IntegralVerbosity supports five predefined SciMLLogging presets:

- **None()** - No output (best for production)
- **Minimal()** - Only deprecation warnings
- **Standard()** - Only deprecation warnings (default, recommended)
- **Detailed()** - Comprehensive output for debugging (all setup, solver, and debug info)
- **All()** - Maximum verbosity (includes iteration-by-iteration progress)

# Usage

## Using Presets

```julia
# Standard preset (default) - shows only deprecations
solve(prob, QuadGKJL(); verbose = IntegralVerbosity())
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(Standard()))

# Completely silent
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(None()))

# Only deprecation warnings
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(Minimal()))

# Detailed debugging information
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(Detailed()))

# Maximum verbosity including iteration progress
solve(prob, QuadGKJL(); verbose = IntegralVerbosity(All()))
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
    buffer_allocation = Silent(),
    deprecations = WarnLevel()
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

# Preset Details

## None
- All toggles: `Silent()`

## Minimal
- `deprecations` → `WarnLevel()` (shows deprecation warnings)
- All others → `Silent()`

## Standard (Default)
- `deprecations` → `WarnLevel()` (shows deprecation warnings)
- All others → `Silent()`

## Detailed
- `cache_init` → `InfoLevel()`
- `domain_transformation` → `InfoLevel()`
- `algorithm_selection` → `InfoLevel()`
- `convergence_result` → `InfoLevel()`
- `batch_mode` → `InfoLevel()`
- `buffer_allocation` → `InfoLevel()`
- `iteration_progress` → `Silent()` (still off, can be very verbose)
- `deprecations` → `WarnLevel()`

## All
- All toggles → `InfoLevel()` (including `iteration_progress`)
- `deprecations` → `WarnLevel()`
"""
function IntegralVerbosity end 

const DEFAULT_VERBOSE = IntegralVerbosity()

@inline function _process_verbose_param(verbose::SciMLLogging.AbstractVerbosityPreset)
    IntegralVerbosity(verbose)
end

@inline function _process_verbose_param(verbose::Bool)
    verbose ? DEFAULT_VERBOSE : IntegralVerbosity(SciMLLogging.None())
end

@inline _process_verbose_param(verbose::IntegralVerbosity) = verbose
