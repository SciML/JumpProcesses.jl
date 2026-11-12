# [How jumps are simulated](@id algorithms_overview)

The pages in this section explain how JumpProcesses simulates jumps, for the
aggregators whose methods go beyond the published algorithms they build on. This
page covers the machinery they all share. The standard stochastic simulation
algorithms (SSAs), such as `Direct`, `NRM` and `RSSA`, follow their references in
[Jumps, JumpProblem, and Aggregators](@ref jump_problem_type).

## Sampling jump times

Consider jumps ``i = 1, \dots, M`` with rates ``\lambda_i(u, p, t)``, where ``u``
is the state, ``p`` the parameters and ``t`` the time. Between jumps the state
may change continuously, through an ODE or SDE, so a rate can vary in time even
while no jump occurs. The total rate is

```math
\Lambda(t) = \sum_{i=1}^{M} \lambda_i\big(u(t), p, t\big).
```

The exact methods in JumpProcesses all rest on one fact. If the last jump was at
time ``t_\ell``, the next jump, of any type, happens at the first time ``\tau``
with

```math
\int_{t_\ell}^{\tau} \Lambda(s)\, ds = E, \qquad E \sim \mathrm{Exp}(1),
```

and it is jump ``i`` with probability ``\lambda_i(\tau) / \Lambda(\tau)``. When
the rates are constant between jumps, as for `ConstantRateJump`s and
`MassActionJump`s, the integral is ``(\tau - t_\ell) \Lambda`` and
``\tau = t_\ell + E / \Lambda``, which is Gillespie's direct method [^1].

The jumps can also be treated separately: jump ``i`` fires when its own
integrated rate, ``\int \lambda_i(s)\, ds``, reaches its own exponential
threshold ``E_i``, and then draws a new one. The thresholds are independent, and
the next jump is the first of these times. This gives the same distribution of
paths, and is the basis of the next reaction methods [^2][^3]. Gillespie's first
reaction method [^1] instead draws new thresholds for all the jumps after each
jump, which is equivalent since exponential thresholds are memoryless. Methods
built on one threshold for the total rate are called *direct* methods.

## Where each jump is handled

[`JumpProblem`](@ref) splits the jumps between two algorithms:

  - The *aggregator*, the positional argument such as `Direct()`, handles
    `MassActionJump`s, `ConstantRateJump`s and, if it supports them, bounded
    `VariableRateJump`s. Currently only `Coevolve` supports them.
  - The *variable-rate aggregator*, chosen with the `vr_aggregator` keyword,
    handles the other `VariableRateJump`s: [`VR_FRM`](@ref) (the default),
    [`VR_Direct`](@ref) or [`VR_DirectFW`](@ref). It needs a continuous problem,
    such as an `ODEProblem`, to step through time.

`RegularJump`s are for τ-leaping methods, which are not exact and are not
covered here.

Both algorithms are callbacks of the time integrator. The aggregator is a
`DiscreteCallback`, and the variable-rate aggregator is one or more
`ContinuousCallback`s or `VectorContinuousCallback`s. The `JumpProblem` stores
them, and every solve initializes them again, drawing new thresholds from the
random number generator passed to `solve` or `init`.

## Aggregators: a discrete callback and stops

An aggregator schedules each jump as a stop of the integrator:

  - When it is initialized, it evaluates the rates, samples the time and type of
    the next jump, and adds that time to the integrator's `tstops`, unless it is
    past the end of the time span.
  - The integrator steps exactly to each of its `tstops`. At the jump time the
    aggregator's condition, that the current time is the next jump time, holds,
    and its affect executes the jump.
  - The affect then updates the rates, samples the next jump, and adds its time
    to the `tstops`. Aggregators with a dependency graph recompute only the rates
    the jump changed.

Since the rates are evaluated only at jumps, they must be constant between jumps.
A `ConstantRateJump`'s rate must not depend on time, on components that evolve
continuously, or on components that `VariableRateJump`s change, unless the
`JumpProblem` is given those components in a [`JumpDependencies`](@ref), which makes
the aggregator update the affected rates after each `VariableRateJump`; see the
[jump diffusion tutorial](@ref jump_diffusion_tutorial). For the same reason, a
callback that changes `u` or `p` must call [`reset_aggregated_jumps!`](@ref), so
that the aggregator recomputes its rates and next jump.

`Coevolve` also handles time-varying rates, through bounds. For a bounded
`VariableRateJump`, it samples candidate times from the upper bound `urate`,
which holds over the window given by `rateinterval`, and accepts a candidate at
time ``s`` with probability ``\lambda(s) / \text{urate}``. This is thinning
[^4][^5]. A rejected candidate, or one past the end of the window, leads to a new
candidate. Each candidate is a stop, so the machinery above applies unchanged.

## Pure jump problems: `SSAStepper`

For a `JumpProblem` over a `DiscreteProblem`, [`SSAStepper`](@ref) steps from
one stop to the next, at jump times and at any user `tstops`, with no
continuous dynamics to integrate. It runs only `DiscreteCallback`s, so it can't
be used with general `VariableRateJump`s.

## Coupling to ODE and SDE solvers

With an ODE, SDE, DDE or DAE problem, the solver steps as it normally would. After
each step from ``t_{\mathrm{prev}}`` to ``t``, it checks the callbacks:

  - **Continuous callbacks.** DiffEqBase evaluates each condition at the step's
    ends, and also at `interp_points` points in between on the solver's
    interpolant. If a condition changes sign, it finds the root on the
    interpolant. The earliest event among all continuous callbacks cuts the step
    there, and its affect runs.
  - **Discrete callbacks.** They are then checked at the end of the step, which
    may have been cut short. The aggregator's condition holds only if the step
    ended at its next jump time. Otherwise that time is still a stop, and a later
    step ends there.

A variable-rate aggregator's condition crosses zero when an integrated rate
reaches its threshold, so the jump time is located on the solver's interpolant.
Its accuracy depends on how accurately the rates are integrated, which is what
the variable-rate aggregators differ in.

## Variable-rate aggregators

  - **`VR_FRM`** treats each jump separately. It extends the state to an
    [`ExtendedJumpArray`](@ref) with one component per jump, which starts at minus
    the jump's threshold and has the jump's rate as its derivative. The solver
    integrates these components along with the rest of the state, so its error
    control covers them, and a component crossing zero is a jump of that type.
    The rates are evaluated whenever the solver evaluates the right-hand side.
    Locating jumps with the solver's event detection follows the hybrid method of
    Salis and Kaznessis [^6].
  - **`VR_Direct`** and **`VR_DirectFW`** use one threshold for the total rate,
    with one `ContinuousCallback`, and leave the state as it is. They integrate
    the total rate over each solver step themselves, then choose which jump
    fires from the rates at the jump time. `VR_DirectFW` stores the rate and
    affect functions in `FunctionWrapper`s, for problems with many jumps. See
    [VR_Direct and VR_DirectFW](@ref vr_direct_algorithm).

|                                  | `VR_FRM`                                  | `VR_Direct`                                      |
|:-------------------------------- |:----------------------------------------- |:------------------------------------------------ |
| State                            | extended by one component per jump        | unchanged                                        |
| Callbacks                        | one per jump, or one vector callback      | one                                              |
| Rates evaluated                  | at each evaluation of the right-hand side | at the points of its fit of each step            |
| Accuracy of the integrated rates | the solver's error control                | its own, with the solver's `abstol` and `reltol` |
| Rates driven by SDE noise        | integrated by the SDE solver              | need steps that resolve the noise                |

## Dependency graphs

When a jump fires, it usually changes only a few components of the state, and
so only a few rates. A dependency graph lists, for each jump, the jumps whose
rates it changes, so that an aggregator recomputes only those. `NRM`, `DirectCR`,
`SortingDirect`, `RSSA`, `RSSACR` and `Coevolve` use one; see
[Jump Aggregators Requiring Dependency Graphs](@ref Jump-Aggregators-Requiring-Dependency-Graphs).
The variable-rate aggregators don't, since the continuous dynamics can change
every rate between jumps.

## References

[^1]: D. T. Gillespie, A general method for numerically simulating the stochastic
    time evolution of coupled chemical reactions, Journal of Computational
    Physics, 22 (4), 403–434 (1976). doi:10.1016/0021-9991(76)90041-3.

[^2]: M. A. Gibson and J. Bruck, Efficient exact stochastic simulation of chemical
    systems with many species and many channels, Journal of Physical Chemistry A,
    104 (9), 1876-1889 (2000). doi:10.1021/jp993732q.

[^3]: D. F. Anderson, A modified next reaction method for simulating chemical
    systems with time dependent propensities and delays, Journal of Chemical
    Physics, 127 (21), 214107 (2007). doi:10.1063/1.2799998.

[^4]: P. A. W. Lewis and G. S. Shedler, Simulation of nonhomogeneous Poisson
    processes by thinning, Naval Research Logistics Quarterly, 26 (3), 403–413
    (1979). doi:10.1002/nav.3800260304.

[^5]: Y. Ogata, On Lewis' simulation method for point processes, IEEE
    Transactions on Information Theory, 27 (1), 23–31 (1981).
    doi:10.1109/TIT.1981.1056305.

[^6]: H. Salis and Y. Kaznessis, Accurate hybrid stochastic simulation of a
    system of coupled chemical or biochemical reactions, Journal of Chemical
    Physics, 122 (5), 054103 (2005). doi:10.1063/1.1835951.

!!! info "Developer notes"

    These notes are for working on JumpProcesses itself.

      - `JumpProblem` builds the aggregator with `aggregate` and wraps it in a
        `DiscreteCallback` (`src/aggregators/ssajump.jl`). The variable-rate
        aggregator's callbacks come from `configure_jump_problem`
        (`src/variable_rate.jl`), which for `VR_FRM` also returns the extended
        problem.
      - The jump callbacks are stored in the `JumpProblem` and reused by every
        solve. Their `initialize` functions reset all of their state, which is why
        sequential solves of one problem are independent but concurrent ones are
        not.
      - All random numbers come from `get_rng(integrator)`, the generator passed
        to `solve` or `init`.
      - In each step, the integrator applies the earliest continuous callback
        event and then the discrete callbacks (`handle_callbacks!` in
        OrdinaryDiffEqCore). DiffEqBase
        evaluates a continuous callback's condition only while checking a step:
        at ``t_{\mathrm{prev}}``, at ``t``, at the interior `interp_points`, and
        during root finding. It is never evaluated after an event or between
        steps. `VR_Direct` relies on this.
