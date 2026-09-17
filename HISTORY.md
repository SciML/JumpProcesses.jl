# Breaking updates and feature summaries across releases

## JumpProcesses unreleased (master branch)

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
