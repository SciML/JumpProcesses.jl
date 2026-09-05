# [Jump Problem and Jump Diffusion Solvers](@id jump_solve)

```julia
solve(prob::JumpProblem, alg; kwargs)
```

## Recommended Methods

`JumpProblem`s can be solved with two classes of methods, exact and inexact.
Exact algorithms currently sample realizations of the jump processes in
chronological order, executing individual jumps sequentially at randomly sampled
times. In contrast, inexact (τ-leaping) methods are time-step based, executing
multiple occurrences of jumps during each time-step. These methods can be much
faster as they only simulate the total number of jumps over each leap interval,
and thus do not need to simulate the realization of every single jump. Jumps for
use with exact simulation methods can be defined as `ConstantRateJump`s,
`MassActionJump`s, and/or `VariableRateJump`. Jumps for use with inexact
τ-leaping methods can be defined as `RegularJump`s or `MassActionJump`s, with
`PureLeaping()` as the common mass-action interface (see the table below).

There are special algorithms available for efficiently simulating an exact, pure
`JumpProblem` (i.e., a `JumpProblem` over a `DiscreteProblem`).  `SSAStepper()`
is an efficient streamlined integrator for time stepping such problems from
individual jump to jump. This integrator is named after Stochastic Simulation
Algorithms (SSAs), commonly used naming in chemistry and biology applications
for the class of exact jump process simulation algorithms. In turn, we denote by
"aggregators" the algorithms that `SSAStepper` calls to calculate the next jump
time and to execute a jump (i.e., change the system state appropriately). All
JumpProcesses aggregators can be used with `ConstantRateJump`s and
`MassActionJump`s, with a subset of aggregators also working with bounded
`VariableRateJump`s (see [the first tutorial](@ref poisson_proc_tutorial) for
the definition of bounded `VariableRateJump`s). Although `SSAStepper()` is
usually faster, it only supports discrete events (`DiscreteCallback`s), for pure
jump problems requiring continuous events (`ContinuousCallback`s) the less
performant `FunctionMap` time-stepper can be used.

For a pure mass-action model, use `SimpleExplicitTauLeaping` for adaptive explicit
leaping, or one of the implicit methods below for stiff systems. For custom
`RegularJump` rates and updates, use `SimpleTauLeaping` for fixed steps or
StochasticDiffEq's `TauLeaping` for adaptive steps. See the
[tau-leaping tutorial](@ref tau_leaping_tutorial) for examples and the
[GPU tutorial](@ref gpu_ensembles) for ensemble execution.

## Special Methods for Pure Jump Problems

If you are using jumps with a differential equation, use the same methods
as in the case of the differential equation solving. However, the following
algorithms are optimized for pure jump problems.

### JumpProcesses.jl

  - `SSAStepper`: a stepping integrator for `JumpProblem`s defined over
    `DiscreteProblem`s involving `ConstantRateJump`s, `MassActionJump`s, and/or
    bounded `VariableRateJump`s . Supports handling of `DiscreteCallback`s and
    saving controls like `saveat`. Note that DifferentialEquations.jl treats
    jumps as similar to callbacks, and hence `SSAStepper` only implements a
    subset of ODE/SDE solver saving controls. In particular, `save_everystep` is
    not supported as saving jumps each step is controlled via the
    `save_positions` argument to [`JumpProblem`](@ref)s. Note, in contrast to
    when [`JumpProblem`](@ref)s are coupled with ODE and SDE timesteppers, with
    [`SSAStepper`](@ref), setting `save_positions = (false, true)`,
    `save_positions = (true, false)` or `save_positions = (true, true)` are
    equivalent and save only after the jump has occurred (as opposed to saving
    the state both before and after a jump). This is because the underlying
    [`SSAStepper`](@ref) generated-solution uses piecewise constant
    interpolation, and can therefore exactly evaluate the sampled solution
    path at any time when only saving the post-jump state for each jump.

## Tau-Leaping Methods

All tau-leaping methods below accept the same
`JumpProblem(prob, PureLeaping(), mass_action_jump)`, with `prob` a
`DiscreteProblem`. Some also accept a `RegularJump` for more general rates and
count-based updates.

| Package | Algorithm | Jump representation | Step selection |
|:--------|:----------|:--------------------|:---------------|
| JumpProcesses | [`SimpleTauLeaping`](@ref) | `MassActionJump` or `RegularJump` | Fixed; requires `dt` |
| JumpProcesses | [`SimpleExplicitTauLeaping`](@ref) | `MassActionJump` | Adaptive explicit; `epsilon` |
| JumpProcesses | [`SimpleImplicitTauLeaping`](@ref) | `MassActionJump` | Adaptive implicit; `epsilon` |
| JumpProcesses | [`SimpleTrapezoidalLeaping`](@ref) | `MassActionJump` | Adaptive implicit trapezoidal; `epsilon` |
| JumpProcesses | [`SimpleAdaptiveTauLeaping`](@ref) | `MassActionJump` | Switches explicit/implicit; `epsilon` |
| StochasticDiffEq | `TauLeaping` | `MassActionJump` or `RegularJump` | Adaptive with post-leap estimates |
| StochasticDiffEq | `CaoTauLeaping` | `MassActionJump` or `RegularJump` | Adaptive Cao step selection |
| StochasticDiffEq | `ImplicitTauLeaping` | `MassActionJump` or `RegularJump` | Implicit; fixed steps with `dt`, `adaptive = false` |
| StochasticDiffEq | `ThetaTrapezoidalTauLeaping` | `MassActionJump` or `RegularJump` | Implicit theta-trapezoidal; fixed steps with `dt`, `adaptive = false` |

### JumpProcesses.jl

The `Simple*` methods listed above are streamlined solvers for pure jump problems:
construct a `DiscreteProblem` and use `PureLeaping()` as the aggregator. The
adaptive mass-action methods require a `MassActionJump`. All methods construct
the rates and stoichiometric updates from it without a user-written
`RegularJump`. They support `saveat`, `save_start`, and `save_end`.

`SimpleImplicitTauLeaping` takes the deterministic part of a leap implicitly.
`SimpleTrapezoidalLeaping` averages current and new-state propensities in that
implicit solve. `SimpleAdaptiveTauLeaping` switches between explicit and implicit
steps based on a stiffness estimate. Its default implicit method is
`SimpleImplicitTauLeaping()`; select the trapezoidal formulation with
`implicit_alg = SimpleTrapezoidalLeaping()`. The default stiffness test uses the
spread of propensities; `eigenvalue_check = true` uses the drift Jacobian instead.
See the algorithm docstrings for the remaining tuning parameters.

Smaller `epsilon` values generally select smaller leaps. This is a step-selection
parameter, not a guarantee of a specified error in an observable or distribution.
Check convergence of the quantities of interest as the leap size is reduced.
Fixed-step `SimpleTauLeaping` does not reject leaps that produce negative
populations, so choosing an appropriate `dt` is especially important.

### StochasticDiffEq.jl

Load `StochasticDiffEq` to use its algorithms. These methods accept a
`MassActionJump` through `PureLeaping()`, or the same
[`RegularJump`](@ref) rate/count-update interface as `SimpleTauLeaping`, with
ordinary differential-equation integrator facilities such as callbacks. When
mixing regular jumps with separately aggregated jumps, select an aggregator such
as `Direct()` for those additional jumps; `PureLeaping()` bypasses aggregation.

The similarly named implicit methods in the two packages are separate algorithms
with different formulations and options. In particular,
`StochasticDiffEq.ImplicitTauLeaping` is not an alias for
`JumpProcesses.SimpleImplicitTauLeaping`, and `ThetaTrapezoidalTauLeaping` is not
`SimpleTrapezoidalLeaping`.

## Regular Jump Diffusion Compatible Methods

Regular jump diffusions are `JumpProblem`s where the internal problem is an `SDEProblem`
and the jump process has designed a regular jump.

### StochasticDiffEq.jl

  - `EM`: Explicit Euler-Maruyama.
  - `ImplicitEM`: Implicit Euler-Maruyama. See the SDE solvers page for more details.
