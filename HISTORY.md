# Breaking updates and feature summaries across releases

## JumpProcesses unreleased (master branch)

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
