# [Migrating to JumpProcesses 10](@id migration_v10)

This guide describes the forthcoming 10.0 API. During integration, the package
still carries 9.x development version metadata; that does not make the changes
below compatible with released 9.x code.

## Move RNG inputs to the solver

`JumpProblem` no longer accepts `rng`. Pass it to `solve` or `init` instead:

```@example migration_rng
using JumpProcesses, Random

maj = MassActionJump([Pair{Int, Int}[]], [[1 => 1]]; param_idxs = [1])
dprob = DiscreteProblem([0], (0.0, 1.0), [2.0])
jprob = JumpProblem(dprob, Direct(), maj)
sol = solve(jprob, SSAStepper(); rng = Xoshiro(1234))
nothing # hide
```

For `SSAStepper`, ODE/DAE solvers, and all five CPU tau-leaping algorithms,
an explicit `rng` is used and advanced without reseeding it. Otherwise, `seed`
creates a fresh `Xoshiro(seed)`; with neither input, the solver uses
`Random.default_rng()`. The CPU tau-leaping methods accept these inputs through
`solve`, not `init`.

StochasticDiffEq uses its own SDE/RODE policy for both `solve` and `init`. An
explicit RNG other than `TaskLocalRNG` takes precedence. With no RNG or a
`TaskLocalRNG`, it uses a nonzero solve/init seed, then the underlying problem's
nonzero seed, then a random seed. It stores a `Xoshiro`, not the `TaskLocalRNG`,
in that case. A stochastic seed of zero means no seed override. See
[Random Number Generator Control](@ref) for the pathway table.

Jump `affect!` functions that draw random numbers should use
`SciMLBase.get_rng(integrator)` to draw from the solver's RNG. Add SciMLBase
as a direct dependency of your project and load it with `using SciMLBase`
before using this interface; JumpProcesses does not re-export `get_rng`.
The old `JumpProcesses.DEFAULT_RNG` constant is removed.

## Keep mass-action definitions separate from working rates

In 9.x, parameter updates could overwrite a `MassActionJump`'s stored scaled
rates. In 10.0, a parameter-mapped definition has `scaled_rates === nothing`.
The solver or aggregator owns a numeric working-rate buffer, filled from the
current parameters at initialization and when the jump state is reset. CPU
tau-leaping solvers fill their buffer at each solve; kernels materialize host
rates before transferring them to the backend.

Use `param_idxs` or a custom `param_mapper` for rates that depend on parameters.
For example, the definition above can be reused with different parameters:

```@example migration_rng
updated = remake(jprob; p = [4.0])
updated_sol = solve(updated, SSAStepper(); seed = 1234)
nothing # hide
```

Definitions constructed with explicit fixed coefficients retain their stored
rates, which are copied into working buffers. Changing `p` does not turn fixed
coefficients into parameter-dependent rates. Definition arrays can still be
mutable; do not mutate shared rate or stoichiometry arrays during solves.

### Constructor and callback keyword changes

Set `scale_rates` and `useiszero` on `MassActionJump`, not `JumpProblem`.
For example, pre-scaled coefficients can be supplied with
`MassActionJump(rates, reactant_stoich, net_stoich; scale_rates = false)`.
Passing either keyword to `JumpProblem` now raises `ArgumentError`.

The old `update_parameters!` mutator is removed, along with the
`update_jump_params` keyword of `reset_aggregated_jumps!`.
After changing parameters or the state in a callback, call
`reset_aggregated_jumps!(integrator)` without that keyword. It refreshes
parameter-mapped working rates as part of reinitializing jump state; parameter
changes during a solve are not detected automatically without a reset.

```@example migration_rng
function change_rate!(integrator)
    integrator.p[1] = 4.0
    reset_aggregated_jumps!(integrator)
    nothing
end
condition(u, t, integrator) = t == 0.5
callback = DiscreteCallback(condition, change_rate!)
callback_sol = solve(remake(jprob; p = [2.0]), SSAStepper();
    seed = 1234, callback, tstops = [0.5])
nothing # hide
```

### Custom parameter mappers

Replace mappers that mutate `maj.scaled_rates` or return a new rate vector with
the in-place callable `(dest, maj, params)`. Fill every entry of `dest`, leave
the definition and input parameters unchanged, and return `nothing`.
`fill_scaled_rates!` delegates to this callable; it does not apply scaling
afterward. The mapper must produce the final scaled coefficients.

For example, this mapper uses a product of two parameters as the rate constant
for a reaction consuming two molecules. It includes the stoichiometric factor
`1 / 2!` directly, so the definition uses `scale_rates = false`:

```@example migration_mapper
using JumpProcesses

function map_rates!(dest, maj, params)
    dest[1] = params[1] * params[2] / 2
    nothing
end

maj = MassActionJump([[1 => 2]], [[1 => -2]];
    param_mapper = map_rates!, scale_rates = false)
dprob = DiscreteProblem([20], (0.0, 1.0), [0.2, 0.5])
jprob = JumpProblem(dprob, Direct(), maj)
sol = solve(jprob, SSAStepper(); seed = 1234)
nothing # hide
```

The built-in `param_idxs` mapper applies stoichiometric scaling according to
`rescale_rates_on_update`, which defaults to `scale_rates`. A custom mapper
that already produces scaled coefficients should use `scale_rates = false`
and avoid scaling them again. Built-in parameter-index mappers can be merged;
merging arbitrary custom mappers is not supported. Combine those reactions in
one definition with one mapper instead.

## Kernel ensemble RNG controls

For multiple trajectories, `EnsembleGPUKernel` uses its backend's RNG and
rejects host `rng` and `rng_func` inputs. `seed` is supported with
`KernelAbstractions.CPU()`. Replay assumes the same thread count, workload and
trajectory count, CPU backend settings, and Julia/package versions. Identical
trajectories across thread counts are not guaranteed.

With a GPU device backend, a non-`nothing` `seed` now raises `ArgumentError`;
previously it was accepted without controlling the device RNG. Seed through
the backend API before `solve`, for example `CUDA.seed!`. These rules apply to
the `SSAStepper`, `SimpleTauLeaping`, and `SimpleExplicitTauLeaping` kernels.
With `trajectories = 1`, execution uses the serial CPU fallback and its usual
RNG/seed controls.

## Jump state ownership, `alias_jump`, and concurrent solves

JumpProcesses 10 never copies a `JumpProblem`'s jump state in the solvers it owns
(`SSAStepper` and the OrdinaryDiffEq ODE/DAE pathways). Every solve and `init`
reuses the problem's jump aggregator and callbacks, re-initializing them from the
current state and parameters. Previously, solves on threads other than thread 1
copied this state.

This makes repeated serial solves cheap, but one `JumpProblem` now supports only
one active solve or integrator at a time. Code that solves one shared problem from
several threads or tasks must give each task its own copy:

```julia
# Before: relied on copies made off thread 1 (no longer made)
Threads.@threads for i in 1:n
    sols[i] = solve(jprob, SSAStepper())
end

# After: copy once per task and reuse the copy within the task
chunks = Iterators.partition(1:n, cld(n, Threads.nthreads()))
tasks = map(chunks) do idxs
    Threads.@spawn begin
        local_prob = deepcopy(jprob)
        [solve(local_prob, SSAStepper()) for _ in idxs]
    end
end
sols = reduce(vcat, fetch.(tasks))
```

An `EnsembleProblem` already provides this isolation: `EnsembleThreads` with
`safetycopy = false` copies the problem once per spawned task, and
`safetycopy = true` copies it before every trajectory. Problems created with
`remake` share the original's jump state, so they are not independent copies;
use `deepcopy` instead. The same applies to integrators from `init` that are
alive at the same time. See the
[ensembles and problem reuse tutorial](@ref ensembles_problem_reuse) for details.

The `alias_jump` keyword argument has been removed, and passing it raises an
error. Delete it from `solve` and `init` calls; `alias_jump = true` is now the
behavior everywhere, and `alias_jump = false` is replaced by solving a copy of
the problem. StochasticDiffEq SDE/RODE solvers keep their own policy, controlled
by `alias = SciMLBase.SDEAliasSpecifier(; alias_jumps = false)` (or
`SciMLBase.RODEAliasSpecifier`), and ignore `alias_jump`.

`SSAStepper` now supports the common `alias` keyword argument for `u0`, `p`, and
`tstops`; see [Jump state ownership and problem reuse](@ref jump_state_ownership).
Its defaults match the previous behavior: `u0` is copied, `p` is reused, and a
caller's `tstops` array is never modified.

Integrator RNGs and separate working-rate buffers do not make a shared
`JumpProblem` safe for concurrent solves on their own. Do not return one shared
mutable RNG from a custom ensemble `rng_func`.

`SortingDirect` retains its learned search order across solves and resets.
For identical seeded trajectories, start from identical learned state as well
as identical model inputs; reusing a problem with a different learned order
can change the trajectory.

## Evaluating an `SSAStepper` integrator at earlier times

`integrator(t)` and `integrator(out, t)` previously returned the current state of
an `SSAStepper` integrator for any `t`. That was silently wrong for times before
the most recent event, which callbacks such as DiffEqCallbacks' `SavingCallback`
and `FunctionCallingCallback` request at their save times. Evaluating at a time
other than `integrator.t` now raises an error, and there are two remedies:

```julia
using DiffEqCallbacks
saved = SavedValues(Float64, Int)
cb = SavingCallback((u, t, integrator) -> sum(u), saved; saveat = 0.0:1.0:10.0)

# Keep the state at the start of each step, at the cost of a copy per step:
sol = solve(jprob, SSAStepper(; save_uprev = true); callback = cb)

# Or stop exactly at the save times, avoiding the per-step copy:
sol = solve(jprob, SSAStepper(); callback = cb, tstops = 0.0:1.0:10.0)
```

`SSAStepper` is now a parametric type, but `SSAStepper()` is unchanged. See
[Saving with callbacks and evaluating the integrator](@ref ssa_integrator_evaluation)
for when to choose each remedy, and [Supported state types](@ref ssa_state_types)
for the state types `SSAStepper` supports.

## Callbacks on the wrapped problem

Callbacks stored on the problem a `JumpProblem` wraps were run by OrdinaryDiffEq
solvers (including `FunctionMap`), but silently ignored by `SSAStepper` and
StochasticDiffEq solvers. They are no longer supported: the `JumpProblem`
constructor, and `remake` with a new wrapped problem, raise an `ArgumentError` if
that problem stores a `callback`. Pass the callbacks to the `JumpProblem`
constructor, or to `solve`/`init`, instead:

```julia
# Before (no longer works):
oprob = ODEProblem(f, u0, tspan, p; callback = cb)
jprob = JumpProblem(oprob, Direct(), jump)

# After:
oprob = ODEProblem(f, u0, tspan, p)
jprob = JumpProblem(oprob, Direct(), jump; callback = cb)
# or pass `callback = cb` to `solve`/`init`.
```

For an existing problem, `remake(oprob; callback = nothing)` removes its callbacks.
