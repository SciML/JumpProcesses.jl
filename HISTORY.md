# Breaking updates and feature summaries across releases

## 10.0 (Breaking)

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
  - RNG state is now owned by the integrator, not the aggregator. This
    eliminates data races when sharing a `JumpProblem` across threads and
    ensures a single, consistent RNG priority across all solver pathways:
    `rng` > `seed` > `Random.default_rng()`.
  - `rng` and `seed` kwargs are fully supported on `solve`/`init` for all
    solver pathways (SSAStepper, ODE, SDE, tau-leaping).
  - `SSAIntegrator` now supports the `SciMLBase` RNG interface (`has_rng`,
    `get_rng`, `set_rng!`).
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
    `param_idxs` or a custom `param_mapper`) are now immutable — rates are
    computed from parameters at aggregator initialization rather than being
    materialized into the jump at `JumpProblem` construction time. This means:
      - `update_parameters!` has been removed. Mass action rates are now
        automatically recomputed from the current parameter values whenever the
        aggregator reinitializes. After modifying parameters (e.g. in a
        callback), call `reset_aggregated_jumps!(integrator)` to trigger
        reinitialization with the updated parameter values.
      - Custom parameter mappers (e.g. ModelingToolkitBase's
        `JumpSysMajParamMapper`) must implement the 3-arg callable API:
        `(mapper)(dest::AbstractVector, maj::MassActionJump, params)`.
        See [`MassActionJumpParamMapper`](@ref) for details.
  - Scalar `param_idxs` (e.g. `param_idxs = 1`) is now internally converted to
    a one-element vector. The scalar form continues to work as before.

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
