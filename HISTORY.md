# Breaking updates and feature summaries across releases

## Unreleased

  - `VR_FRM` now keeps `SciMLBase.FullSpecialize` and `SciMLBase.NoSpecialize` when it
    rebuilds the problem's function. Previously every specialization became
    `AutoSpecialize`. With `FullSpecialize`, `VR_FRM` solves allocate less and are often
    faster, but each new model takes longer to compile.
  - `VR_FRM` no longer slows down with more than 32 `VariableRateJump`s. Its rate
    evaluation recursed over the jumps, which Julia only infers for up to 32 elements, so
    with more jumps every rate evaluation was dynamically dispatched and allocated.
    `VariableRateJump`s passed as a `Vector`, for example through
    `JumpSet(; variable_jumps)`, are now converted to a `Tuple` once, when the
    `JumpProblem` is built, so their rates are evaluated the same type-stable way.
  - Under `SciMLBase.AutoDespecialize`, the integrator passes the jumps a
    `SciMLBase.DespecializedParameters`. The aggregators that store rate functions in
    `FunctionWrapper`s now throw an `ArgumentError` saying to use `FullSpecialize`,
    instead of a `MethodError` at the first rate evaluation.
  - The SciMLBase compat lower bound is now 3.46.

## 10.0 (Breaking)

JumpProcesses 10.0 is a transitional release. StochasticDiffEq, ModelingToolkit, and
Catalyst do not yet support it, so environments that include them keep JumpProcesses
9.x until their updates, and JumpProcesses 10.0.1, are released. Jump problems coupled
to delay differential equations (DDEs) are not supported in JumpProcesses 10.0 at this
time.

  - **Breaking**: The `rng` keyword argument has been removed from
    `JumpProblem`. Pass `rng` to `solve` or `init` instead:
    ```julia
    # Before (no longer works):
    jprob = JumpProblem(dprob, Direct(), jump; rng = Xoshiro(1234))
    sol = solve(jprob, SSAStepper())
    # After:
    jprob = JumpProblem(dprob, Direct(), jump)
    sol = solve(jprob, SSAStepper(); rng = Xoshiro(1234))
    ```
  - **Breaking**: `JumpProcesses.DEFAULT_RNG` has been removed. Custom jump
    affects should draw from `SciMLBase.get_rng(integrator)`. Add SciMLBase as
    a direct project dependency and load it with `using SciMLBase` to use this
    interface. Pass `rng` or `seed` to `solve`/`init` to control the solver's RNG.
  - Jump callbacks now sample from the integrator's RNG. `JumpProblem`s,
    aggregators, and callbacks no longer store their own RNGs. Aggregators and
    callbacks still contain mutable working state, so concurrent solves must
    isolate that state; sharing an arbitrary `JumpProblem` across threads or
    tasks is not made safe by this change.
  - `SSAStepper` and the OrdinaryDiffEq ODE/DAE pathways accept `rng` and `seed`
    through `solve`/`init`; all five CPU tau-leaping algorithms accept them
    through `solve`. These pathways use `rng` first, then `Xoshiro(seed)` if a
    seed is supplied, then `Random.default_rng()`.
  - StochasticDiffEq SDE/RODE `solve` and `init` follow the backend's RNG policy.
    An explicit RNG other than `TaskLocalRNG` takes priority over seeds. With
    no RNG or with `TaskLocalRNG`, a nonzero solve/init seed takes priority,
    followed by the underlying problem's nonzero seed, then a randomly seeded
    `Xoshiro`. `TaskLocalRNG` is converted to `Xoshiro`; `seed = 0` means no seed
    override on this pathway, whereas the JumpProcesses RNG resolver uses
    `Xoshiro(0)`.
  - **Breaking**: Solvers owned by JumpProcesses (`SSAStepper` and the
    OrdinaryDiffEq ODE/DAE pathways) no longer copy a `JumpProblem`'s jump state.
    On every thread they reuse the problem's aggregator and jump callbacks,
    re-initializing them at each `init`; previously, solves on threads other than
    thread 1 copied this state. A `JumpProblem` therefore supports one active solve
    or integrator at a time. Concurrent solves from threads or tasks, and
    integrators alive at the same time, must each use an independent copy, such as
    `deepcopy(jprob)`. Problems created with `remake` share the original's jump
    state, so they are not independent either. `EnsembleProblem` provides the
    needed copies: `EnsembleThreads` copies the problem once per spawned task, and
    `safetycopy = true` copies it for every trajectory. See the new tutorial on
    ensembles and problem reuse for patterns that balance performance and safety.
  - **Breaking**: The `alias_jump` keyword argument has been removed. Passing it to
    `solve`, `init`, `JumpProblem`, or a multi-trajectory `EnsembleGPUKernel`
    solve, or storing it on the problem wrapped by a `JumpProblem`, raises an
    error; remove it, and solve a copy of the problem when
    independent jump state is needed. StochasticDiffEq SDE/RODE solvers ignore the
    keyword and keep their own policy: pass
    `alias = SciMLBase.SDEAliasSpecifier(; alias_jumps = false)` (or
    `SciMLBase.RODEAliasSpecifier` for RODEs) to copy jump state, or
    `alias_jumps = true` to reuse it. When unspecified, those solvers reuse jump
    state on thread 1 and copy it on other threads.
  - **Breaking**: Callbacks stored on the problem wrapped by a `JumpProblem` are not
    supported. The `JumpProblem` constructor, and `remake` with a new wrapped problem,
    now raise an `ArgumentError` if that problem stores a `callback` (other than
    `nothing` or an empty `CallbackSet`). Previously, OrdinaryDiffEq solvers
    (including `FunctionMap`) ran such callbacks, while `SSAStepper` and
    StochasticDiffEq SDE/RODE solvers silently ignored them. Pass callbacks to the
    `JumpProblem` constructor or to `solve`/`init` instead:
    ```julia
    # Before (no longer works):
    oprob = ODEProblem(f, u0, tspan, p; callback = cb)
    jprob = JumpProblem(oprob, Direct(), jump)
    # After:
    oprob = ODEProblem(f, u0, tspan, p)
    jprob = JumpProblem(oprob, Direct(), jump; callback = cb)
    ```
    `remake(prob; callback = nothing)` removes callbacks from an existing problem.
  - `SSAStepper` now supports the common `alias` keyword argument for its `u0`, `p`,
    and `tstops` inputs, given as a `Bool`, a `SciMLBase.DiscreteAliasSpecifier`, or
    an `ODEAliasSpecifier` (which also controls `tstops`). The defaults match
    OrdinaryDiffEq: `u0` is copied, `p` is reused, and a caller's `tstops` array is
    never modified. `alias` never controls jump state.
  - `SSAStepper` now saves states, and evaluates `integrator(t)`, with
    `recursivecopy` as OrdinaryDiffEq does, so nested state types are saved as
    independent snapshots. `tstops` given as a number, a tuple, or an
    `AbstractVector` other than a `Vector` of the time type (such as a range or
    view) is copied into a `Vector` at `init`, so `add_tstop!` works with it.
  - **Breaking**: Evaluating an `SSAStepper` integrator, as `integrator(t)` or
    `integrator(out, t)`, at a time other than `integrator.t` now raises an error,
    unless the problem is solved with `SSAStepper(; save_uprev = true)`, which also
    allows times in the current step, `[integrator.tprev, integrator.t]`.
    Previously the current state was returned for any `t`, which was silently wrong
    for times before the most recent event, for example in a `SavingCallback` or
    `FunctionCallingCallback` with save times. Alternatively, pass such times as
    `tstops`, so that the integrator stops exactly at them. DiffEqCallbacks'
    integrating callbacks are not yet supported with `SSAStepper`.
  - New `SSAStepper(; save_uprev = true)` option: the integrator keeps the state at
    the start of each step in `integrator.uprev`, copying the state on every step,
    and provides `get_tmp_cache` for callbacks that evaluate it in place. The
    default, `SSAStepper()`, keeps no copy. `SSAStepper` is now a parametric type;
    `SSAStepper()` is unchanged and has type `SSAStepper{false}`. After `solve!`,
    `integrator.tprev` is the time of the last event or tstop.
  - Fixed `SSAStepper` saving pending `saveat` times before the end time after its
    final callback pass. A callback that changed the state at the end time altered the
    values saved at those earlier times, and the saved times came out of order. They
    are now saved before the final callback pass.
  - Fixed `SSAStepper` erroring with a `BoundsError` when saving only the final state:
    `save_end = true` with nothing saved earlier, for example with `save_start = false`,
    `save_positions = (false, false)`, and no `saveat`. The final state is now saved.
  - Fixed `init` on ODE, DAE, and `FunctionMap` integrators for `JumpProblem`s:
    callbacks stored in the `JumpProblem` (`JumpProblem(...; callback)`) were merged
    twice, so they ran twice per step, and were reintroduced even with
    `merge_callbacks = false`. `init` now combines stored and call-level callbacks
    exactly as `solve` does.
  - Stochastic `init` of a `JumpProblem` now reaches StochasticDiffEqCore's own
    per-algorithm `JumpProblem` initializers, so `init` followed by `solve!` matches
    `solve`. This requires a StochasticDiffEqCore release that supports JumpProcesses 10.
  - The state types `SSAStepper` supports are now documented: vectors and
    `SVector`s of integers or floats, species × sites matrices for spatial
    problems, and scalars for models without mass-action jumps. See "Supported
    state types" in the solver documentation.
  - **Breaking**: Multi-trajectory `EnsembleGPUKernel` solves reject `alias` values
    other than `nothing` and the `alias_jumps` keyword, whether passed to `solve` or
    stored on the problem, since kernels always build device-owned state. They
    also reject `SSAStepper(; save_uprev = true)`, since kernels never expose an
    integrator.
  - The minimum supported SciMLBase version is now 3.34.
  - `SSAIntegrator` now supports the `SciMLBase` RNG interface (`has_rng`,
    `get_rng`, `set_rng!`).
  - **Breaking**: Passing `seed` to a multi-trajectory `EnsembleGPUKernel`
    solve with a GPU device backend now raises `ArgumentError`, instead of
    accepting a seed that does not control the device RNG. This applies to
    `SSAStepper`, `SimpleTauLeaping`, and `SimpleExplicitTauLeaping`. Seed the
    device through the backend's API before solving (for example,
    `CUDA.seed!`). Multi-trajectory kernel solves also reject host `rng`
    objects and `rng_func`. The `CPU()` backend still supports `seed` using
    Julia's task-local RNG; repeatability requires the same thread count,
    workload/trajectory count, CPU backend settings, and Julia/package versions.
    Identical trajectories across thread counts are not guaranteed. With `trajectories = 1`,
    `EnsembleGPUKernel` retains its serial CPU fallback and that pathway's
    usual RNG/seed rules.
  - **Breaking**: The `scale_rates` and `useiszero` keyword arguments have been
    removed from `JumpProblem`. Set them on the `MassActionJump` directly:
    ```julia
    # Before (no longer works):
    jprob = JumpProblem(dprob, Direct(), maj; scale_rates = false)
    # After:
    maj = MassActionJump(rates, reactant_stoch, net_stoch; scale_rates = false)
    jprob = JumpProblem(dprob, Direct(), maj)
    ```
  - **Breaking**: Parameterized `MassActionJump`s (those constructed with
    `param_idxs` or a custom `param_mapper`) retain `scaled_rates === nothing`.
    Solvers and aggregators own the working rate buffers and fill them from
    current parameters at initialization and reset. `remake` no longer mutates
    shared rate coefficients in the jump definition. Fixed-rate and symbolic
    definitions retain their supported `scaled_rates` representation. This means:
      + `update_parameters!` has been removed. Mass action rates are now
        automatically recomputed from the current parameter values whenever the
        aggregator reinitializes. After modifying parameters (e.g. in a
        callback), call `reset_aggregated_jumps!(integrator)` to trigger
        reinitialization with the updated parameter values.
      + The `update_jump_params` keyword has been removed from
        `reset_aggregated_jumps!`; supplying either `true` or `false` raises an
        explanatory error. Rate refresh is automatic during reset.
      + Custom parameter mappers (e.g. ModelingToolkitBase's
        `JumpSysMajParamMapper`) must replace the old extraction/mutation
        callables with `(mapper)(dest::AbstractVector, maj::MassActionJump, params)`
        and fill `dest` with the scaled rates. The built-in mapper honors
        `maj.rescale_rates_on_update`; custom mappers own their scaling policy
        and must avoid scaling already-scaled symbolic expressions twice.
        See the [custom mapper migration guide](docs/src/migration.md#custom-parameter-mappers)
        for an example.
  - Scalar `param_idxs` (e.g. `param_idxs = 1`) is now internally converted to
    a one-element vector. The scalar form continues to work as before.

## 9.33.1

  - Fixed `SSAStepper` saving the wrong state at the final requested `saveat`
    time when it precedes a later jump. The observation now contains the state
    at the requested time, and output times remain sorted when jump times are
    also saved.

## 9.33

  - Added user-specified rate bounds for `ConstantRateJump`s, enabling rates that
    are non-monotonic or increase in some species and decrease in others to be
    used with `RSSA` and `RSSACR` ([#653](https://github.com/SciML/JumpProcesses.jl/pull/653)).
    Pass `bounds(ulow, uhigh, u, p, t)` through the `bounds` keyword when
    constructing the jump. Here `ulow` and `uhigh` define the current species
    population bracket, and `u`, `p`, and `t` are the current state, parameters,
    and time. The function returns `RateBounds(; lrate, urate)` with nonnegative,
    finite bounds satisfying `lrate <= rate(v, p, t) <= urate` for every state
    `v` in that bracket. These bounds must hold throughout the bracket, not just
    at the current state. As with all `ConstantRateJump`s, the rate must remain
    constant between jumps and must not explicitly depend on time.

    For example, the following jump converts one particle of species 1 into
    species 2 at a rate that increases with species 1 and decreases with species 2:

    ```julia
    using JumpProcesses
    rate(u, p, t) = p[1] * u[1] / (1 + u[2])
    function affect!(integrator)
        integrator.u[1] -= 1
        integrator.u[2] += 1
        nothing
    end
    function bounds(ulow, uhigh, u, p, t)
        RateBounds(
            lrate = p[1] * ulow[1] / (1 + uhigh[2]),
            urate = p[1] * uhigh[1] / (1 + ulow[2])
        )
    end
    jump = ConstantRateJump(rate, affect!; bounds)
    prob = DiscreteProblem([10, 0], (0.0, 10.0), [1.0])
    # Each species affects the rate of jump 1; jump 1 changes both species.
    vartojumps_map = [[1], [1]]
    jumptovars_map = [[1, 2]]
    jprob = JumpProblem(prob, RSSA(), jump; vartojumps_map, jumptovars_map)
    sol = solve(jprob, SSAStepper())
    ```

    The same example works with `RSSACR()` in place of `RSSA()`. Omitting
    `bounds` preserves the existing behavior: rate bounds are computed by
    evaluating the rate at `ulow` and `uhigh`. This is valid for rates that are
    nondecreasing in every species or nonincreasing in every species, but is not
    generally valid for mixed dependence such as the example above.

## 9.14

  - Added the constant complexity next reaction method (CCNRM).

## 9.13

  - Added a default aggregator selection algorithm based on the number of passed
    in jumps. i.e. the following now auto-selects an aggregator (`Direct` in this
    case):

    ```julia
    using JumpProcesses
    rate(u, p, t) = u[1]
    affect(integrator) = (integrator.u[1] -= 1; nothing)
    crj = ConstantRateJump(rate, affect)
    dprob = DiscreteProblem([10], (0.0, 10.0))
    jprob = JumpProblem(dprob, crj)
    sol = solve(jprob, SSAStepper())
    ```

  - For `JumpProblem`s over `DiscreteProblem`s that only have `MassActionJump`s,
    `ConstantRateJump`s, and bounded `VariableRateJump`s, one no longer needs to
    specify `SSAStepper()` when calling `solve`, i.e. the following now works for
    the previous example and is equivalent to manually passing `SSAStepper()`:

    ```julia
    sol = solve(jprob)
    ```

  - Plotting a solution generated with `save_positions = (false, false)` now uses
    piecewise linear plots between any saved time points specified via `saveat`
    instead (previously the plots appeared piecewise constant even though each
    jump was not being shown). Note that solution objects still use piecewise
    constant interpolation, see [the
    docs](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/#save_positions_docs)
    for details.

## 9.7

  - `Coevolve` was updated to support use with coupled ODEs/SDEs. See the updated
    documentation for details, and note the comments there about one needing to ensure
    rate bounds hold however the ODE/SDE stepper could modify dependent variables during a timestep.

## 9.3

  - Support for "bounded" `VariableRateJump`s that can be used with the `Coevolve`
    aggregator for faster simulation of jump processes with time-dependent rates.
    In particular, if all `VariableRateJump`s in a pure-jump system are bounded one
    can use `Coevolve` with `SSAStepper` for better performance. See the
    documentation, particularly the first and second tutorials, for details on
    defining and using bounded `VariableRateJump`s.
