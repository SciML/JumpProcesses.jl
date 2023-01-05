# [Jump Problem and Jump Diffusion Solvers](@id jump_solve)

```julia
solve(prob::JumpProblem,alg;kwargs)
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
τ-leaping methods should be defined as `RegularJump`s.

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

If there is a `RegularJump`, then inexact τ-leaping methods must be used. The
current recommended method is `TauLeaping` if one needs adaptivity, events, etc.
If one only needs the most barebones fixed time-step leaping method, then
`SimpleTauLeaping` can have performance benefits.

## Special Methods for Pure Jump Problems

If you are using jumps with a differential equation, use the same methods
as in the case of the differential equation solving. However, the following
algorithms are optimized for pure jump problems.

### JumpProcesses.jl

- `SSAStepper`: a stepping integrator for `JumpProblem`s defined over
  `DiscreteProblem`s involving `ConstantRateJump`s, `MassActionJump`s, and/or
  bounded `VariableRateJump`s . Supports handling of `DiscreteCallback`s and
  saving controls like `saveat`.

## RegularJump Compatible Methods

### StochasticDiffEq.jl

These methods support mixing with event handling, other jump types, and all of
the features of the normal differential equation solvers.

- `TauLeaping`: an adaptive tau-leaping algorithm with post-leap estimates.

### JumpProcesses.jl

- `SimpleTauLeaping`: a tau-leaping algorithm for pure `RegularJump` `JumpProblem`s.
  Requires a choice of `dt`.
- `RegularSSA`: a version of SSA for pure `RegularJump` `JumpProblem`s.

## Regular Jump Diffusion Compatible Methods

Regular jump diffusions are `JumpProblem`s where the internal problem is an `SDEProblem`
and the jump process has designed a regular jump.

### StochasticDiffEq.jl

- `EM`: Explicit Euler-Maruyama.
- `ImplicitEM`: Implicit Euler-Maruyama. See the SDE solvers page for more details.
