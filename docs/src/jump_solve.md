# [Jump Problem and Jump Diffusion Solvers](@id jump_solve)

```julia
solve(prob::JumpProblem,alg;kwargs)
```

## Recommended Methods

Because `JumpProblem`s can be solved with two classes of methods, exact and
inexact, they come in two forms. Exact algorithms tend to describe the
realization of each jump chronologically. Alternatively, inexact methods tend to
take small leaps through time so they are guaranteed to terminate in finite
time. These methods can be much faster as they only simulate the total number of
points in each leap interval and thus do not need to simulate the realization of
every single jump. Jumps for exact methods can be defined with
`ConstantRateJump`, `VariableRateJump` and/or `MassActionJump`  On the other
hand, jumps for inexact methods are defined with `RegularJump`.

There are special algorithms available for a pure exact `JumpProblem` (a
`JumpProblem` over a  `DiscreteProblem`).  The `SSAStepper()` is an efficient
streamlined integrator for running simulation algorithms of such problems. This
integrator is named after the term Stochastic Simulation Algorithm (SSA) which
is a catch-all term in biochemistry to denote algorithms for simulating jump
processes. In turn, we denote aggregators algorithms for simulating jump
processes that can use the `SSAStepper()` integrator. These algorithms can solve
problems initialized with `ConstantRateJump`, `VariableRateJump` and/or
`MassActionJump`.  Although `SSAStepper()` is usually faster, it is not
compatible with event handling. If events are necessary, then `FunctionMap` does
well.

If there is a `RegularJump`, then inexact methods must be used. The current
recommended method is `TauLeaping` if you need adaptivity, events, etc. If you
just need the most barebones fixed time step leaping method, then
`SimpleTauLeaping` can have performance benefits.

## Special Methods for Pure Jump Problems

If you are using jumps with a differential equation, use the same methods
as in the case of the differential equation solving. However, the following
algorithms are optimized for pure jump problems.

### JumpProcesses.jl

- `SSAStepper`: a stepping integrator for pure `ConstantRateJump`,
  `VariableRateJump` and/or `MassActionJump` `JumpProblem`s. Supports handling
  of `DiscreteCallback` and saving controls like `saveat`.

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
