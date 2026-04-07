# JumpProcesses.jl API

```@meta
CurrentModule = JumpProcesses
```

## Core Types

```@docs
JumpProblem
SSAStepper
reset_aggregated_jumps!
```

## Jump Types

```@docs
ConstantRateJump
MassActionJump
VariableRateJump
RegularJump
JumpSet
```

## Aggregator Types

Aggregators are the underlying algorithms used for sampling
[`ConstantRateJump`](@ref)s, [`MassActionJump`](@ref)s, and
[`VariableRateJump`](@ref)s.

```@docs
Coevolve
Direct
DirectCR
FRM
NRM
RDirect
RSSA
RSSACR
SortingDirect
```

## Random Number Generator Control

JumpProcesses supports controlling the random number generator (RNG) used for
jump sampling via the `rng` and `seed` keyword arguments to `solve` or `init`.

### `rng` keyword argument

Pass any `AbstractRNG` to `solve` or `init`:

```julia
using Random, StableRNGs

# Using a StableRNG for cross-version reproducibility
sol = solve(jprob, SSAStepper(); rng = StableRNG(1234))

# Using Julia's built-in Xoshiro
sol = solve(jprob, Tsit5(); rng = Xoshiro(42))
```

### `seed` keyword argument

As a shorthand, pass an integer `seed` to create a `Xoshiro` generator:

```julia
sol = solve(jprob, SSAStepper(); seed = 1234)
# equivalent to: solve(jprob, SSAStepper(); rng = Xoshiro(1234))
```

### Resolution priority

When both `rng` and `seed` are passed to the same `solve`/`init` call, `rng`
takes priority:

| User provides | Result |
|---|---|
| `rng` via `solve`/`init` | Uses that `rng` |
| `seed` via `solve`/`init` | Creates `Xoshiro(seed)` |
| Nothing | Uses `Random.default_rng()` (SSAStepper, ODE, tau-leaping) or a randomly-seeded `Xoshiro` (SDE) |

### Behavior by solver pathway

| Solver | Default RNG (nothing passed) | `rng` / `seed` support |
|---|---|---|
| `SSAStepper` | `Random.default_rng()` | Full support via `solve`/`init` kwargs |
| ODE solvers (e.g., `Tsit5`) | `Random.default_rng()` | Full support via `solve`/`init` kwargs |
| SDE solvers (e.g., `SRIW1`) | Randomly-seeded `Xoshiro` | Full support; `TaskLocalRNG` is auto-converted to `Xoshiro` |
| `SimpleTauLeaping` | `Random.default_rng()` | Full support via `solve` kwargs |

!!! note
    For reproducible simulations, always pass an explicit `rng` or `seed`.
    The default RNG is shared global state and may produce different results
    depending on prior usage.

# Private / Developer API

```@docs
ExtendedJumpArray
SSAIntegrator
```

## Internal Dispatch Pathways

The following table documents which code handles `solve`/`init` for each solver
type. This is relevant for developers working on JumpProcesses or its solver
backends.

| Solver type | `__solve` handled by | `__init` handled by | Uses `__jump_init`? |
|---|---|---|---|
| `SSAStepper` | JumpProcesses (`solve.jl`) | JumpProcesses (`SSA_stepper.jl`) | No |
| ODE (e.g., `Tsit5`) | JumpProcesses (`solve.jl`) | JumpProcesses (`solve.jl`) → OrdinaryDiffEq | Yes |
| SDE (e.g., `SRIW1`) | StochasticDiffEq | StochasticDiffEq | No |
| `SimpleTauLeaping` | JumpProcesses (`simple_regular_solve.jl`, custom `DiffEqBase.solve`) | N/A | No |

For **SSAStepper**, `rng` is resolved via `resolve_rng` in `SSA_stepper.jl`'s
`__init` and stored on the [`SSAIntegrator`](@ref).

For **ODE solvers**, `rng` is resolved via `resolve_rng` in `__jump_init`
(`solve.jl`) and forwarded to OrdinaryDiffEq's `init`, which stores it on the
`ODEIntegrator`.

For **SDE solvers**, StochasticDiffEq handles the full solve/init pathway
directly (JumpProcesses' ambiguity-fix `__solve` method is never dispatched to).
StochasticDiffEq has its own `_resolve_rng` that additionally handles
`TaskLocalRNG` conversion and the problem's stored seed.

For **tau-leaping**, JumpProcesses defines a custom `DiffEqBase.solve` that
bypasses the standard `__solve`/`__init` pathway. It calls `resolve_rng`
directly with the `rng` and `seed` kwargs from the `solve` call.
