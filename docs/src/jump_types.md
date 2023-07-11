# [Jump Problems](@id jump_problem_type)

## Mathematical Specification of a problem with jumps

Jumps (or point) processes are stochastic processes with discrete state changes
driven by a `rate` function. The homogeneous Poisson process is the canonical
point process with a constant rate of change. Processes involving multiple jumps
are known as compound jump (or point) processes.

A compound Poisson process is a continuous-time Markov Chain where the time to
the next jump is exponentially distributed as determined by the rate. Simulation
algorithms for these types of processes are known in biology and chemistry as
Gillespie methods or Stochastic Simulation Algorithms (SSA), with the time
evolution that the probability these processes are in a given state at a given
time satisfying the Chemical Master Equation (CME). In the statistics literature,
the composition of Poisson processes is described by the superposition theorem.

Any differential equation can be extended by jumps. For example, we have an ODE
with jumps, denoted by

```math
\frac{du}{dt} = f(u,p,t) + \sum_{i}c_i(u,p,t)p_i(t)
```

where ``p_i`` is a Poisson counter of rate ``\lambda_i(u,p,t)``. Extending
a stochastic differential equation to have jumps is commonly known as a Jump
Diffusion, and is denoted by

```math
du(t) = f(u,p,t)dt + \sum_{j}g_j(u,t)dW_j(t) + \sum_{i}c_i(u,p,t)dp_i(t)
```

## Types of Jumps: Constant Rate, Mass Action, Variable Rate and Regular

Exact jump process simulation algorithms tend to describe the realization of
each jump event chronologically. Individual jumps are usually associated with
changes to the state variable `u`, which in turn changes the `rate`s at which
jump events occur. These jumps can be specified as a [`ConstantRateJump`](@ref),
[`MassActionJump`](@ref), or a [`VariableRateJump`](@ref).

Each individual type of jump that can occur is represented through (implicitly
or explicitly) specifying two pieces of information; a `rate` function (i.e.,
intensity or propensity) for the jump and an `affect!` function for the jump.
The former gives the probability per time a particular jump can occur given the
current state of the system, and hence determines the time at which jumps can
happen. The latter specifies the instantaneous change in the state of the system
when the jump occurs.

A specific jump type is a [`VariableRateJump`](@ref) if its rate function is
dependent on values which may change between the occurrence of any two jump
events of the process. Examples include jumps where the rate is an explicit
function of time, or depends on a state variable that is modified via continuous
dynamics such as an ODE or SDE. Such "general" `VariableRateJump`s can be
expensive to simulate because it is necessary to consider the (possibly
continuous) changes in the rate function when calculating the next jump time.

*Bounded* [`VariableRateJump`](@ref)s represent a special subset of
`VariableRateJump`s where one can specify functions that calculate a time window
over which the rate is bounded by a constant (presuming the state `u` is
unchanged due to another `ConstantRateJump`, `MassActionJump` or bounded
`VariableRateJump`). They can be simulated more efficiently using
rejection-sampling based approaches that leverage this upper bound.

[`ConstantRateJump`](@ref)s are more restricted in that they assume the rate
functions are constant at all times between two consecutive jumps of the system.
That is, any states or parameters that a rate function depends on must not
change between the times at which two consecutive jumps occur.

A [`MassActionJump`](@ref)s is a specialized representation for a collection of
`ConstantRateJump` jumps that can each be interpreted as a standard mass action
reaction. For systems comprised of many mass action reactions, using the
`MassActionJump` type will offer improved performance compared to using
multiple `ConstantRateJump`s. Note, only one `MassActionJump` should be defined
per [`JumpProblem`](@ref); it is then responsible for handling all mass action
reaction type jumps. For systems with both mass action jumps and non-mass action
jumps, one can create one `MassActionJump` to handle the mass action jumps, and
create a number of `ConstantRateJump`s or `VariableRateJump`s to handle the
non-mass action jumps.

Since exact methods simulate each individual jump, they may become
computationally expensive to simulate processes over timescales that involve
*many* jump occurrences. As an alternative, inexact τ-leaping methods take
discrete steps through time, over which they simultaneously execute many jumps.
These methods can be much faster as they do not need to simulate the realization
of every individual jump event. τ-leaping methods trade accuracy for speed, and
are best used when a set of jumps do not make significant changes to the
processes' state and/or rates over the course of one time-step (i.e., during a
leap interval). A single [`RegularJump`](@ref) is used to encode jumps for
τ-leaping algorithms. While τ-leaping methods can be proven to converge in the
limit that the time-step approaches zero, their accuracy can be highly dependent
on the chosen time-step. As a rule of thumb, if changes to the state variable
`u` during a time-step (i.e., leap interval) are "minimal" compared to the size of
the system, an τ-leaping method can often provide reasonable solution
approximations.

Currently, `ConstantRateJump`s, `MassActionJump`s, and `VariableRateJump`s can
be coupled to standard SciML ODE/SDE solvers since they are internally handled
via callbacks. For `ConstantRateJump`s, `MassActionJump`s, and bounded
`VariableRateJump` the determination of the next jump time and type is handled
by a user-selected *aggregator* algorithm. `RegularJump`s currently require
their own special time integrators.

#### Defining a Constant Rate Jump

The constructor for a [`ConstantRateJump`](@ref) is:

```julia
ConstantRateJump(rate, affect!)
```

  - `rate(u, p, t)` is a function which calculates the rate given the current
    state `u`, parameters `p`, and time `t`.
  - `affect!(integrator)` is the effect on the equation using the integrator
    interface. It encodes how the state should change due to *one* occurrence of
    the jump.

#### Defining a Mass Action Jump

The constructor for a [`MassActionJump`](@ref) is:

```julia
MassActionJump(reactant_stoich, net_stoich; scale_rates = true, param_idxs = nothing)
```

  - `reactant_stoich` is a vector whose `k`th entry is the reactant stoichiometry
    of the `k`th reaction. The reactant stoichiometry for an individual reaction
    is assumed to be represented as a vector of `Pair`s, mapping species integer
    id to stoichiometric coefficient.
  - `net_stoich` is assumed to have the same type as `reactant_stoich`; a
    vector whose `k`th entry is the net stoichiometry of the `k`th reaction. The
    net stoichiometry for an individual reaction is again represented as a vector
    of `Pair`s, mapping species id to the net change in the species when the
    reaction occurs.
  - `scale_rates` is an optional parameter that specifies whether the rate
    constants correspond to stochastic rate constants in the sense used by
    Gillespie, and hence need to be rescaled. *The default, `scale_rates = true`,
    corresponds to rescaling the passed in rate constants.* See below.
  - `param_idxs` is a vector of the indices within the parameter vector, `p`, that
    correspond to the rate constant for each jump.

**Notes for Mass Action Jumps**

  - When using `MassActionJump` the default behavior is to assume rate constants
    correspond to stochastic rate constants in the sense used by Gillespie (J.
    Comp. Phys., 1976, 22 (4)). This means that for a reaction such as ``2A \overset{k}{\rightarrow} B``, the jump rate function constructed by
    `MassActionJump` would be `k*A*(A-1)/2!`. For a trimolecular reaction like
    ``3A \overset{k}{\rightarrow} B`` the rate function would be
    `k*A*(A-1)*(A-2)/3!`. To *avoid* having the reaction rates rescaled (by `1/2`
    and `1/6` for these two examples), one can pass the `MassActionJump`
    constructor the optional named parameter `scale_rates = false`, i.e., use
    
    ```julia
    MassActionJump(reactant_stoich, net_stoich; scale_rates = false, param_idxs)
    ```

  - Zero order reactions can be passed as `reactant_stoich`s in one of two ways.
    Consider the ``\varnothing \overset{k}{\rightarrow} A`` reaction with rate
    `k=1`:
    
    ```julia
    p = [1.0]
    reactant_stoich = [[0 => 1]]
    net_stoich = [[1 => 1]]
    jump = MassActionJump(reactant_stoich, net_stoich; param_idxs = [1])
    ```
    
    Alternatively, one can create an empty vector of pairs to represent the reaction:
    
    ```julia
    p = [1.0]
    reactant_stoich = [Vector{Pair{Int, Int}}()]
    net_stoich = [[1 => 1]]
    jump = MassActionJump(reactant_stoich, net_stoich; param_idxs = [1])
    ```
  - For performance reasons, it is recommended to order species indices in
    stoichiometry vectors from smallest to largest. That is
    
    ```julia
    reactant_stoich = [[1 => 2, 3 => 1, 4 => 2], [2 => 2, 3 => 2]]
    ```
    
    is preferred over
    
    ```julia
    reactant_stoich = [[3 => 1, 1 => 2, 4 => 2], [3 => 2, 2 => 2]]
    ```

#### Defining a Variable Rate Jump

The constructor for a [`VariableRateJump`](@ref) is:

```julia
VariableRateJump(rate, affect!;
                 lrate = nothing, urate = nothing, rateinterval = nothing,
                 idxs = nothing, rootfind = true, save_positions = (true, true),
                 interp_points = 10, abstol = 1e-12, reltol = 0)
```

  - `rate(u, p, t)` is a function which calculates the rate given the current
    state `u`, parameters `p`, and time `t`.
  - `affect!(integrator)` is the effect on the equation using the integrator
    interface. It encodes how the state should change due to *one* occurrence of
    the jump.

To define a bounded `VariableRateJump`, which can be simulated more efficiently
with bounded `VariableRateJump` supporting aggregators such as `Coevolve`, one
must also specify

  - `urate(u, p, t)`, a function which computes an upper bound for the rate in the
    interval `t` to `t + rateinterval(u, p, t)` at time `t` given state `u` and
    parameters `p`.
  - `rateinterval(u, p, t)`, a function which computes a time interval `t` to `t + rateinterval(u, p, t)` given state `u` and parameters `p` over which the
    `urate` bound will hold (and `lrate` bound if provided, see below).

For increased performance, one can also specify a lower bound that should be
valid over the same `rateinterval`

  - `lrate(u, p, t)`, a function which computes a lower bound for the rate in the
    interval `t` to `t + rateinterval(u, p, t)` at time `t` given state `u` and
    parameters `p`. `lrate` should remain valid under the same conditions as
    `urate`.

Note that

  - It is currently only possible to simulate `VariableRateJump`s with
    `SSAStepper` when using systems with only bounded `VariableRateJump`s and the
    `Coevolve` aggregator.
  - When coupling `Coevolve` with a continuous problem type such as an
    `ODEProblem` ensure that the bounds are satisfied given changes in `u` over
    `rateinterval`. `Coevolve` handles jumps in the same way whether it is
    using the `SSAStepper` or other continuous steppers.
  - On the other hand, when choosing a different aggregator than `Coevolve`,
    `SSAStepper` cannot currently be used, and the `JumpProblem` must be
    coupled to a continuous problem type such as an `ODEProblem` to handle
    time-stepping. The continuous time-stepper treats *all* `VariableRateJump`s
    as `ContinuousCallback`s, using the `rate(u, p, t)` function to construct
    the `condition` function that triggers a callback.

#### Defining a Regular Jump

The constructor for a [`RegularJump`](@ref) is:

```julia
RegularJump(rate, c, numjumps; mark_dist = nothing)
```

  - `rate(out, u, p, t)` is the function which computes the rate for every regular
    jump process
  - `c(du, u, p, t, counts, mark)` calculates the update given `counts` number of
    jumps for each jump process in the interval.
  - `numjumps` is the number of jump processes, i.e., the number of `rate`
    equations and the number of `counts`.
  - `mark_dist` is the distribution for a mark.

## Defining a Jump Problem

To define a `JumpProblem`, one must first define the basic problem. This can be
a `DiscreteProblem` if there is no differential equation, or an ODE/SDE/DDE/DAE
if you would like to augment a differential equation with jumps. Denote this
previously defined problem as `prob`. Then the constructor for the jump problem
is:

```julia
JumpProblem(prob, aggregator, jumps::JumpSet;
            save_positions = typeof(prob) <: AbstractDiscreteProblem ? (false, true) :
                             (true, true))
```

The aggregator is the method for simulating `ConstantRateJump`s,
`MassActionJump`s, and bounded `VariableRateJump`s (if supported by the
aggregator). They are called aggregators since they resolve all these jumps in a
single discrete simulation algorithm. The possible aggregators are given below.
`jumps` is a [`JumpSet`](@ref) which is just a collection of jumps. Instead of
passing a `JumpSet`, one may just pass a list of jumps as trailing positional
arguments. For example:

```julia
JumpProblem(prob, aggregator, jump1, jump2)
```

and the internals will automatically build the `JumpSet`. `save_positions`
determines whether to save the state of the system just before and/or after
jumps occur.

Note that a `JumpProblem`/`JumpSet` can only have 1 `RegularJump` (since a
`RegularJump` itself describes multiple processes together). Similarly, it can
only have one `MassActionJump` (since it also describes multiple processes
together).

## Jump Aggregators for Exact Simulation

Jump aggregators are methods for simulating `ConstantRateJump`s,
`MassActionJump`s, and bounded `VariableRateJump`s (if supported) exactly. They
are called aggregators since they combine all jumps to handle within a single
discrete simulation algorithm. Aggregators combine jumps in different ways and
offer different trade-offs. However, all aggregators describe the realization of
each and every individual jump chronologically. Since they do not skip any
jumps, they are considered exact methods. Note that none of the aggregators
discussed in this section can be used with `RegularJumps` which are used for
time-step based (inexact) τ-leaping methods.

The current aggregators are (note that an italicized name indicates the
aggregator requires various types of dependency graphs, see the next section):

  - `Direct`: The Gillespie Direct method SSA [1].
  - `DirectFW`: the Gillespie Direct method SSA [1] with `FunctionWrappers`. This
    aggregator uses a different internal storage format for collections of
    `ConstantRateJumps`.
  - *`DirectCR`*: The Composition-Rejection Direct method of Slepoy et al [2]. For
    large networks and linear chain-type networks, it will often give better
    performance than `Direct`.
  - *`SortingDirect`*: The Sorting Direct Method of McCollum et al [3]. It will
    usually offer performance as good as `Direct`, and for some systems can offer
    substantially better performance.
  - *`RSSA`*: The Rejection SSA (RSSA) method of Thanh et al [4,5]. With `RSSACR`,
    for very large reaction networks, it often offers the best performance of all
    methods.
  - *`RSSACR`*: The Rejection SSA (RSSA) with Composition-Rejection method of
    Thanh et al [6]. With `RSSA`, for very large reaction networks, it often offers
    the best performance of all methods.
  - `RDirect`: A variant of Gillespie's Direct method [1] that uses rejection to
    sample the next reaction.
  - `FRM`: The Gillespie first reaction method SSA [1]. `Direct` should generally
    offer better performance and be preferred to `FRM`.
  - `FRMFW`: The Gillespie first reaction method SSA [1] with `FunctionWrappers`.
  - *`NRM`*: The Gibson-Bruck Next Reaction Method [7]. For some reaction network
    structures, this may offer better performance than `Direct` (for example,
    large, linear chains of reactions).
  - *`Coevolve`*: An improvement of the COEVOLVE algorithm of Farajtabar et al
    [8]. Currently the only aggregator that also supports *bounded*
    `VariableRateJump`s. As opposed
    to COEVOLVE, this method syncs the thinning procedure with the stepper
    which allows it to handle dependencies on continuous dynamics. Essentially
    reduces to `NRM` in handling `ConstantRateJump`s and `MassActionJump`s.

To pass the aggregator, pass the instantiation of the type. For example:

```julia
JumpProblem(prob, Direct(), jump1, jump2)
```

will build a problem where the jumps are simulated using Gillespie's Direct SSA
method.

[1] Daniel T. Gillespie, A general method for numerically simulating the stochastic
time evolution of coupled chemical reactions, Journal of Computational Physics,
22 (4), 403–434 (1976). doi:10.1016/0021-9991(76)90041-3.

[2] A. Slepoy, A.P. Thompson and S.J. Plimpton, A constant-time kinetic Monte
Carlo algorithm for simulation of large biochemical reaction networks, Journal
of Chemical Physics, 128 (20), 205101 (2008). doi:10.1063/1.2919546.

[3] J. M. McCollum, G. D. Peterson, C. D. Cox, M. L. Simpson and N. F.
Samatova, The sorting direct method for stochastic simulation of biochemical
systems with varying reaction execution behavior, Computational Biology and
Chemistry, 30 (1), 39049 (2006). doi:10.1016/j.compbiolchem.2005.10.007.

[4] V. H. Thanh, C. Priami and R. Zunino, Efficient rejection-based simulation
of biochemical reactions with stochastic noise and delays, Journal of Chemical
Physics, 141 (13), 134116 (2014). doi:10.1063/1.4896985.

[5] V. H. Thanh, R. Zunino and C. Priami, On the rejection-based algorithm for
simulation and analysis of large-scale reaction networks, Journal of Chemical
Physics, 142 (24), 244106 (2015). doi:10.1063/1.4922923.

[6] V. H. Thanh, R. Zunino, and C. Priami, Efficient constant-time complexity
algorithm for stochastic simulation of large reaction networks, IEEE/ACM
Transactions on Computational Biology and Bioinformatics, 14 (3), 657-667
(2017). doi:10.1109/TCBB.2016.2530066.

[7] M. A. Gibson and J. Bruck, Efficient exact stochastic simulation of chemical
systems with many species and many channels, Journal of Physical Chemistry A,
104 (9), 1876-1889 (2000). doi:10.1021/jp993732q.

[8] M. Farajtabar, Y. Wang, M. Gomez-Rodriguez, S. Li, H. Zha, and L. Song,
COEVOLVE: a joint point process model for information diffusion and network
evolution, Journal of Machine Learning Research 18(1), 1305–1353 (2017). doi:
10.5555/3122009.3122050.

## Jump Aggregators Requiring Dependency Graphs

Italicized constant rate jump aggregators above require the user to pass a
dependency graph to `JumpProblem`. `Coevolve`, `DirectCR`, `NRM`, and
`SortingDirect` require a jump-jump dependency graph, passed through the named
parameter `dep_graph`. i.e.,

```julia
JumpProblem(prob, DirectCR(), jump1, jump2; dep_graph = your_dependency_graph)
```

For systems with only `MassActionJump`s, or those generated from a
[Catalyst](https://docs.sciml.ai/Catalyst/stable/) `reaction_network`, this
graph will be auto-generated. Otherwise, you must construct the dependency graph
whenever the set of jumps include `ConstantRateJump`s and/or bounded
`VariableRateJump`s.

Dependency graphs are represented as a `Vector{Vector{Int}}`, with the `i`th
vector containing the indices of the jumps for which rates must be recalculated
when the `i`th jump occurs. Internally, all `MassActionJump`s are ordered before
`ConstantRateJump`s and bounded `VariableRateJump`s. General `VariableRateJump`s
are not handled by aggregators, and so not included in the jump ordering for
dependency graphs. Note that the relative order between `ConstantRateJump`s and
relative order between bounded `VariableRateJump`s is preserved. In this way, one
can precalculate the jump order to manually construct dependency graphs.

`RSSA` and `RSSACR` require two different types of dependency graphs, passed
through the following `JumpProblem` kwargs:

  - `vartojumps_map` - A `Vector{Vector{Int}}` mapping each variable index, `i`,
    to a set of jump indices. The jump indices correspond to jumps with rate
    functions that depend on the value of `u[i]`.
  - `jumptovars_map` - A `Vector{Vector{Int}}`  mapping each jump index to a set
    of variable indices. The corresponding variables are those that have their
    value, `u[i]`, altered when the jump occurs.

For systems generated from a [Catalyst](https://docs.sciml.ai/Catalyst/stable/)
`reaction_network` these will be auto-generated. Otherwise, you must explicitly
construct and pass in these mappings.

## Recommendations for exact methods

For representing and aggregating jumps

  - Use a `MassActionJump` to handle all jumps that can be represented as mass
    action reactions with constant rate between jumps. This will generally offer
    the fastest performance.
  - Use `ConstantRateJump`s for any remaining jumps with a constant rate between
    jumps.
  - Use `VariableRateJump`s for any remaining jumps with variable rate between
    jumps. If possible, construct a bounded [`VariableRateJump`](@ref) as
    described above and in the doc string. The tighter and easier to compute
    the bounds are, the faster the resulting simulation will be. Use the
    `Coevolve` aggregator to ensure such jumps are handled via the more
    efficient aggregator interface. `Coevolve` handles continuous steppers so
    can be coupled with a continuous problem type such as an `ODEProblem` as
    long as the bounds are satisfied given changes in `u` over `rateinterval`.

For systems with only `ConstantRateJump`s and `MassActionJump`s,

  - For a small number of jumps, < ~10, `Direct` will often perform as well as the
    other aggregators.
  - For > ~10 jumps `SortingDirect` will often offer better performance than
    `Direct`.
  - For large numbers of jumps with sparse chain like structures and similar jump
    rates, for example continuous time random walks, `RSSACR`, `DirectCR` and then
    `NRM` often have the best performance.
  - For very large networks, with many updates per jump, `RSSA` and `RSSACR` will
    often substantially outperform the other methods.

For pure jump systems, time-step using `SSAStepper()` with a `DiscreteProblem`
unless one has general (i.e., non-bounded) `VariableRateJump`s.

In general, for systems with sparse dependency graphs if `Direct` is slow, one
of `SortingDirect`, `RSSA` or `RSSACR` will usually offer substantially better
performance. See
[DiffEqBenchmarks.jl](https://docs.sciml.ai/SciMLBenchmarksOutput/stable/) for
benchmarks on several example networks.

## Remaking `JumpProblem`s

When running many simulations, it can often be convenient to update the initial
condition or simulation parameters without having to create and initialize a new
`JumpProblem`. In such situations `remake` can be used to change the initial
condition, time span, and the parameter vector. **Note,** the new `JumpProblem`
will alias internal data structures from the old problem, including core
components of the SSA aggregators. As such, only the new problem generated by
`remake` should be used for subsequent simulations.

As an example, consider the following SIR model:

```julia
rate1(u, p, t) = p[1] * u[1] * u[2]
function affect1!(integrator)
    integrator.u[1] -= 1
    integrator.u[2] += 1
end
jump = ConstantRateJump(rate1, affect1!)

rate2(u, p, t) = p[2] * u[2]
function affect2!(integrator)
    integrator.u[2] -= 1
    integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate2, affect2!)
u0 = [999, 1, 0]
p = (0.1 / 1000, 0.01)
tspan = (0.0, 250.0)
dprob = DiscreteProblem(u0, tspan, p)
jprob = JumpProblem(dprob, Direct(), jump, jump2)
sol = solve(jprob, SSAStepper())
```

We can change any of `u0`, `p` and/or `tspan` by either making a new
`DiscreteProblem`

```julia
u02 = [10, 1, 0]
p2 = (0.1 / 1000, 0.0)
tspan2 = (0.0, 2500.0)
dprob2 = DiscreteProblem(u02, tspan2, p2)
jprob2 = remake(jprob, prob = dprob2)
sol2 = solve(jprob2, SSAStepper())
```

or by directly remaking with the new parameters

```julia
jprob2 = remake(jprob, u0 = u02, p = p2, tspan = tspan2)
sol2 = solve(jprob2, SSAStepper())
```

To avoid ambiguities, the following will give an error

```julia
jprob2 = remake(jprob, prob = dprob2, u0 = u02)
```

as will trying to update either `p` or `tspan` while passing a new
`DiscreteProblem` using the `prob` kwarg.
