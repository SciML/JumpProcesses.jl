# JumpProcesses.jl API

```@meta
CurrentModule = JumpProcesses
```

## Core Types

```@docs
ExtendedJumpArray
JumpProblem
PureLeaping
SSAStepper
SplitCoupledJumpProblem
reset_aggregated_jumps!
remake(::JumpProblem)
```

## Jump Types

```@docs
ConstantRateJump
MassActionJump
VariableRateJump
RegularJump
JumpSet
RateBounds
```

## Aggregator Types

Aggregators are the underlying algorithms used for sampling
[`ConstantRateJump`](@ref)s, [`MassActionJump`](@ref)s, and
[`VariableRateJump`](@ref)s.

```@docs
BracketData
CCNRM
Coevolve
Direct
DirectCR
DirectCRDirect
DirectFW
FRM
FRMFW
NRM
NSM
RDirect
RSSA
RSSACR
SortingDirect
get_num_majumps
needs_depgraph
needs_vartojumps_map
```

## Variable Rate Aggregators

```@docs
VariableRateAggregator
VR_Direct
VR_DirectFW
VR_FRM
```

## Tau-Leaping Algorithms

```@docs
EnsembleGPUKernel
SimpleAdaptiveTauLeaping
SimpleExplicitTauLeaping
SimpleImplicitTauLeaping
SimpleTauLeaping
SimpleTrapezoidalLeaping
```

## Spatial Jump APIs

```@docs
CartesianGrid
CartesianGridRej
SpatialMassActionJump
neighbors
num_sites
outdegree
```

## Reexported SciML common interface

`using JumpProcesses` also brings in the parts of the SciML common interface needed to
build the problem a [`JumpProblem`](@ref) wraps, solve it, drive the integrator from a
jump's `affect!`, and inspect the result -- so they do not have to be imported
separately. These names are owned and documented by
[SciMLBase](https://docs.sciml.ai/SciMLBase/stable/); JumpProcesses only re-exports
them:

  - Problems: [`DiscreteProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/),
    [`ODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/),
    [`SDEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/sde_types/),
    [`EnsembleProblem`](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/),
    `remake`, `NullParameters`
  - Functions: `DiscreteFunction`, `ODEFunction`, `SDEFunction`
  - Solutions: [`ODESolution`](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/),
    `EnsembleSolution`, `EnsembleSummary`, and the `EnsembleAnalysis` module
  - Ensemble algorithms: `EnsembleSerial`, `EnsembleThreads`, `EnsembleDistributed`,
    `EnsembleSplitThreads`
  - Solving: `solve`, `solve!`, `init`, `step!`
  - Integrator interface: `add_tstop!`, `add_saveat!`, `savevalues!`,
    `set_proposed_dt!`, `set_t!`, `set_u!`, `reinit!`, `terminate!`, `u_modified!`,
    `derivative_discontinuity!`
  - Return status: `ReturnCode`, `successful_retcode`
  - [Callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/):
    `DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback`, `CallbackSet`

`DiscreteProblem` and `EnsembleProblem` in particular are what most downstream code
reaches for through JumpProcesses -- see SciML/MomentClosure.jl#111 for what happens
when they are not re-exported.

Note that [`SSAStepper`](@ref) only supports `DiscreteCallback`s;
`ContinuousCallback` and `VectorContinuousCallback` are re-exported for use with the
ODE/SDE integrators a `JumpProblem` can be paired with.

Anything else from SciMLBase -- the BVP, DAE, DDE, nonlinear and optimization problem
classes, the SciML operators, and the internals -- is not re-exported here; import it
from SciMLBase directly.

## [Jump state ownership and problem reuse](@id jump_state_ownership)

A [`JumpProblem`](@ref) stores mutable jump state: the jump aggregator and its
callbacks. Solvers owned by JumpProcesses ([`SSAStepper`](@ref) and the
OrdinaryDiffEq ODE/DAE pathways) reuse this state directly and re-initialize it
from the current state and parameters at every `init`. They never copy it.
Consequently:

  - Repeated serial solves of one problem are safe and avoid any copying.
  - One `JumpProblem` supports one active solve or integrator at a time.
    Concurrent solves from threads or tasks, and integrators that are alive at the
    same time, each need an independent copy, such as `deepcopy(jprob)`.
  - [`remake`](@ref remake(::JumpProblem)) shares the original's jump state, so
    remade problems are not independent for concurrent use.
  - `EnsembleProblem` supplies isolation. `EnsembleSerial` reuses the problem
    sequentially. `EnsembleThreads` with `safetycopy = false` copies the problem
    once per spawned task, with a serial fallback for one thread or a single
    trajectory. `safetycopy = true` copies the problem before every trajectory,
    before `prob_func` runs.

The [ensembles and problem reuse tutorial](@ref ensembles_problem_reuse) shows
patterns that balance performance and safety, including ensemble hooks, stateful
callbacks, manual threading, and distributed ensembles.

The `alias_jump` keyword argument has been removed; passing it raises an error.
The plural `alias_jumps` is not a solver keyword argument either: `solve` and
`init` reject it through SciMLBase's keyword validation, and multi-trajectory
`EnsembleGPUKernel` solves, which bypass that validation, reject it explicitly.

### Aliasing `SSAStepper` inputs

[`SSAStepper`](@ref) supports the common SciML `alias` keyword argument for its
`u0`, `p`, and `tstops` inputs. Pass a `Bool` to apply one choice to all three, a
`SciMLBase.DiscreteAliasSpecifier` to control `u0` and `p`, or an
`ODEAliasSpecifier` to also control `tstops`. A field set to `true` permits
aliasing, `false` requests a copy, and `nothing` selects the default:

| Input    | Default (`nothing`)                  | `true`                                               | `false`                     |
|:-------- |:------------------------------------ |:---------------------------------------------------- |:--------------------------- |
| `u0`     | copied with `recursivecopy`          | the integrator uses and mutates `u0`                 | copied                      |
| `p`      | reused                               | reused                                               | copied with `recursivecopy` |
| `tstops` | the caller's array is never modified | new stops may be inserted into the caller's `Vector` | copied at `init`            |

`tstops` may be a number, a tuple, or an `AbstractVector` of times (such as a
range or view), or a callable `(p, tspan) -> times`. Only a `Vector` of the time
type can be aliased; other supported inputs are copied into a new `Vector` at
`init`. `alias` never controls jump state; for example,
`alias = false` does not give a solve its own jump aggregator.

```julia
sol = solve(jprob, SSAStepper(); alias = SciMLBase.DiscreteAliasSpecifier(alias_p = false))
```

### Coupled ODE/DAE problems

For a `JumpProblem` wrapping an ODE or DAE problem, `alias` is passed unchanged to
the underlying solver. OrdinaryDiffEq solvers, including its DAE solvers, require
an `ODEAliasSpecifier`. The jump state is always reused, as described above.

### StochasticDiffEq SDE/RODE problems

StochasticDiffEq owns jump-state copying for SDE and RODE problems. Pass
`alias = SciMLBase.SDEAliasSpecifier(; alias_jumps = false)`, or
`SciMLBase.RODEAliasSpecifier` for RODE problems, to copy the jump state, or set
`alias_jumps = true` to reuse it. When the field is unspecified, the backend
reuses the jump state on thread 1 and copies it on other threads. This backend
ignores the removed `alias_jump` keyword.

### Kernel ensembles

Multi-trajectory [`EnsembleGPUKernel`](@ref) solves always build device-owned
state. They reject the removed `alias_jump` keyword and any `alias` value other
than `nothing`, whether passed to `solve` or stored on the problem wrapped by the
`JumpProblem`. With `trajectories = 1`, the serial solver's rules apply.

## Random Number Generator Control

JumpProcesses supports controlling the random number generator (RNG) used for
jump sampling via the `rng` and `seed` keyword arguments to `solve` or `init`,
as supported by the solver pathway below. `JumpProblem` no longer accepts `rng`;
see [Migrating to JumpProcesses 10](@ref migration_v10).

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

For SSAStepper, OrdinaryDiffEq ODE/DAE solvers, and all five CPU tau-leaping
algorithms, an explicit `rng` takes priority over `seed`:

| User provides             | Result                      |
|:------------------------- |:--------------------------- |
| `rng` via `solve`/`init`  | Uses that `rng`             |
| `seed` via `solve`/`init` | Creates `Xoshiro(seed)`     |
| Nothing                   | Uses `Random.default_rng()` |

StochasticDiffEq handles SDE/RODE RNGs for both `solve` and `init`. An explicit
RNG other than `TaskLocalRNG` takes priority over seeds. With no RNG, or with
`TaskLocalRNG`, a nonzero solve/init `seed` takes priority, followed by the
underlying problem's nonzero `seed`; otherwise StochasticDiffEq creates a
randomly seeded `Xoshiro`. It converts `TaskLocalRNG` to `Xoshiro` rather than
storing it on the stochastic integrator. In this pathway, `seed=0` means no
seed override, whereas SSAStepper, the ODE/DAE pathway, and CPU tau-leaping
solvers use `Xoshiro(0)`.

### Behavior by solver pathway

| Solver                                                                                                         | Default RNG (nothing passed)                             | `rng` / `seed` support                                                                   |
|:-------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------------- |:---------------------------------------------------------------------------------------- |
| `SSAStepper`                                                                                                   | `Random.default_rng()`                                   | Full support via `solve`/`init` kwargs                                                   |
| OrdinaryDiffEq ODE/DAE solvers (e.g., `Tsit5`, `DFBDF`)                                                        | `Random.default_rng()`                                   | Full support via `solve`/`init` kwargs                                                   |
| StochasticDiffEq SDE/RODE solvers (e.g., `SRIW1`, `RandomEM`)                                                  | `Xoshiro` from the stored problem seed, or a random seed | Full support; `TaskLocalRNG` follows the seed policy above and is converted to `Xoshiro` |
| `SimpleTauLeaping`                                                                                             | `Random.default_rng()`                                   | Full support via `solve` kwargs                                                          |
| `SimpleExplicitTauLeaping`, `SimpleImplicitTauLeaping`, `SimpleTrapezoidalLeaping`, `SimpleAdaptiveTauLeaping` | `Random.default_rng()`                                   | Full support via `solve` kwargs                                                          |

!!! note

    Use an explicit `rng` or `seed` where the solver supports it, and start from
    identical model and mutable aggregator state. An explicit RNG is advanced
    by the solve; construct or copy it from the same starting state for replay.
    Julia's default RNG is task-local; results may depend on prior draws in the task.
    `SortingDirect` retains its learned search order, which can also affect replay.

### Kernel ensembles

Multi-trajectory [`EnsembleGPUKernel`](@ref) solves use backend-local RNGs and
reject host `rng` and `rng_func` inputs. With `KernelAbstractions.CPU()`, `seed`
seeds Julia's task-local RNG. Replay assumes the same thread count, workload
and trajectory count, CPU backend settings, and Julia/package versions;
identical trajectories across thread counts are not guaranteed. This differs
from CPU ensemble methods that assign RNGs explicitly to each trajectory.

GPU device backends reject `seed` with `ArgumentError`; seed the backend
before solving, for example with `CUDA.seed!`. With `trajectories = 1`, the
serial CPU fallback accepts its usual RNG/seed inputs.

# Private / Developer API

```@docs
SSAIntegrator
```

## Internal Dispatch Pathways

The following table documents which code handles `solve`/`init` for each solver
type. This is relevant for developers working on JumpProcesses or its solver
backends.

| Solver type                                           | `__solve` handled by                                                 | `__init` handled by                                           | Uses `__jump_init`? |
|:----------------------------------------------------- |:-------------------------------------------------------------------- |:------------------------------------------------------------- |:------------------- |
| `SSAStepper`                                          | JumpProcesses (`solve.jl`)                                           | JumpProcesses (`SSA_stepper.jl`)                              | No                  |
| OrdinaryDiffEq ODE/DAE (e.g., `Tsit5`, `DFBDF`)       | JumpProcesses (`solve.jl`)                                           | JumpProcesses' OrdinaryDiffEqCore extension → OrdinaryDiffEq  | Yes                 |
| StochasticDiffEq SDE/RODE (e.g., `SRIW1`, `RandomEM`) | StochasticDiffEqCore                                                 | StochasticDiffEqCore's JumpProblem initializer (see below)    | No                  |
| StochasticDiffEq jump algorithms (e.g., `TauLeaping`) | StochasticDiffEqCore                                                 | StochasticDiffEqCore's specialized jump-algorithm initializer | No                  |
| All five CPU tau-leaping algorithms                   | JumpProcesses (`simple_regular_solve.jl`, custom `DiffEqBase.solve`) | N/A                                                           | No                  |

For **SSAStepper**, `rng` is resolved via `resolve_rng` in `SSA_stepper.jl`'s
`__init` and stored on the [`SSAIntegrator`](@ref).

For **OrdinaryDiffEq ODE/DAE solvers**, `rng` is resolved via `resolve_rng` in `__jump_init`
(`solve.jl`) and forwarded to OrdinaryDiffEq's `init`, which stores it on the
`ODEIntegrator`.

For **StochasticDiffEq SDE/RODE solvers**, both `solve` and `init` reach the
backend's JumpProblem initializer, StochasticDiffEqCore's `_sde_init`, with the
original JumpProblem and unchanged RNG/seed inputs, so they share its RNG policy,
jump-state copying, and callback setup. `solve` goes through the backend's more
specific `__solve`. For `init`, the backend's per-algorithm `__init(::JumpProblem, ...)`
methods, and its method for jump algorithms such as `TauLeaping`, are more specific than
JumpProcesses' OrdinaryDiffEqCore extension method and own dispatch directly.

DiffEqBase's `init_call` merges the JumpProblem's stored keywords and user callbacks
once, before dispatching to `__init`; JumpProcesses' `__init` methods do not merge
them again, and discard `merge_callbacks`. The `solve` path merges once in `__solve`.
Only the `__jump_init` pathways (OrdinaryDiffEq ODE/DAE solvers and `FunctionMap`)
read keywords stored on the wrapped problem, through its own `init`; `SSAStepper` and
StochasticDiffEqCore's `_sde_init` never do. So that every solver runs the same
callbacks, the `JumpProblem` constructor, and `remake` with a new wrapped problem,
reject a wrapped problem that stores a `callback`.

Jump-state copying on these pathways follows the backend's alias specifier; see
[Jump state ownership and problem reuse](@ref jump_state_ownership). In contrast,
`SSAStepper` and `__jump_init` never copy jump state: they build the integrator
around the problem's own jump callbacks, which `init` re-initializes. `__jump_init`
raises the removed-`alias_jump` error and forwards `alias` unchanged to the
underlying solver. JumpProcesses still defines the transitional
`resetted_jump_problem` and `reset_jump_problem!` helpers for released
StochasticDiffEq versions, but no longer calls them itself.

For **tau-leaping**, JumpProcesses defines a custom `DiffEqBase.solve` that
bypasses the standard `__solve`/`__init` pathway. It calls `resolve_rng`
directly with the `rng` and `seed` kwargs from the `solve` call.
