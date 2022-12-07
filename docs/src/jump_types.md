# [Jump Problems](@id jump_problem_type)

### Mathematical Specification of an problem with jumps

Jumps (or point) processes are stochastic processes with discrete changes driven
by some `rate`. The homogeneous Poisson process is the canonical point process
with a constant rate of change. Processes involving multiple jumps are known as
compound jump (or point) processes. 

A compound Poisson process is a continuous-time Markov Chain where the time to
the next jump is exponentially distributed as calculated by the rate. This type
of process is known in biology as "Gillespie discrete stochastic simulation",
modeled by the Chemical Master Equation (CME). Alternatively, in the statistics
literature the composition of Poisson processes is described by the
superposition theorem.

Any differential equation can be extended by jumps. For example, we have an ODE
with jumps, denoted by

```math
\frac{du}{dt} = f(u,p,t) + \sum_{i}c_i(u,p,t)p_i(t)
```

where ``p_i`` is a Poisson counter of rate ``\lambda_i(u,p,t)``. Extending
a stochastic differential equation to have jumps is commonly known as a Jump
Diffusion, and is denoted by

```math
du = f(u,p,t)dt + \sum_{j}g_j(u,t)dW_j(t) + \sum_{i}c_i(u,p,t)dp_i(t)
```

## Types of Jumps: Regular, Variable, Constant Rate and Mass Action


Exact algorithms tend to describe the realization of each jump chronologically. 
In more complex cases, such jumps are conditioned on the history of past events. 
Such jumps are usually associated with changes to the state variable `u` which 
in turn changes the `rate` of event occurence. These jumps can be specified as 
a `ConstantRateJump`, `MassActionJump`, or a `VariableRateJump`. Since exact 
methods simulate each and every point, they might face termination issues when 
the `rate` of event occurrence explodes.

Alternatively, inexact methods tend to take small leaps through time so they are 
guaranteed to terminate in finite time. These methods can be much faster as they 
only simulate the total number of points in each leap interval and thus do not 
need to simulate the realization of every single jump. Since inexact methods 
trade accuracy for speed, they should be used when a set of jumps do not make 
significant changes to the system during the leap interval. A `RegularJump` is 
used for inexact algorithms. Note that inexact methods are not always 
inaccurate. In the case of homogeneous Poisson processes, they produce accurate 
results. However, they can produce less accurate results for more complex 
problems, thus it is important to have a good understanding of the problem. As 
a rule of thumb, if changes to the state variable `u` during a leap are minimal 
compared to size of the system, an inexact method should provide reasonable 
solutions.

We denote a jump as variable if its rate function is dependent on values
which may change between any jump in the system. For instance, when the rate is 
a function of time. Variable jumps can be more expensive to simulate because it 
is necessary to take into account the dynamics of the rate function when 
simulating the next jump time.

A `MassActionJump` is a specialized representation for a collection of constant
rate jumps that can each be interpreted as a standard mass action reaction. For
systems comprised of many mass action reactions, using the `MassActionJump` type
will offer improved performance. Note, only one `MassActionJump` should be
defined per `JumpProblem`; it is then responsible for handling all mass action
reaction type jumps. For systems with both mass action jumps and non-mass action
jumps, one can create one `MassActionJump` to handle the mass action jumps, and
create a number of `ConstantRateJumps` or `VariableRateJump` to handle the 
non-mass action jumps.

`RegularJump`s are optimized for inexact jumping algorithms like tau-leaping and
hybrid algorithms. `ConstantRateJump`, `VariableRateJump`, `MassActionJump` are
optimized for exact methods (also known in the biochemistry literature as SSA
algorithms). `ConstantRateJump`s, `VariableRateJump`s and `MassActionJump`s can
be added to standard DiffEq algorithms since they are simply callbacks, while
`RegularJump`s require special algorithms.

#### Defining a Constant Rate Jump

The constructor for a `ConstantRateJump` is:

```julia
ConstantRateJump(rate,affect!)
```

- `rate(u,p,t)` is a function which calculates the rate given the time and the state.
- `affect!(integrator)` is the effect on the equation, using the integrator interface.

#### Defining a Variable Rate Jump

The constructor for a `VariableRateJump` is:

```julia
VariableRateJump(rate,affect!; 
                 lrate=nothing, urate=nothing, L=nothing
                 idxs=nothing,
                 rootfind=true,
                 save_positions=(true,true),
                 interp_points=10,
                 abstol=1e-12,reltol=0)
```

- `rate(u,p,t)` is a function which calculates the rate given the time and the state.
- `affect!(integrator)` is the effect on the equation, using the integrator interface.
- When planning to use the `QueueMethod` aggregator, the arguments `lrate`,
  `urate` and `L` are required. They consist of three functions: `lrate(u, p,
  t)` computes the lower bound of the intensity rate in the interval `t` to `t 
  + L` given state `u` and parameters `p`; `urate(u, p, t)` computes the upper
  bound of the intensity rate; and `L(u, p, t)` computes the interval length
  for which the rate is bounded between `lrate` and `urate`. 
- It is only possible to solve a `VariableRateJump` with `SSAStepper` when using 
  the `QueueMethod` aggregator.
- When using a different aggregator than `QueueMethod`, there is no need to 
  define `lrate`, `urate` and `L`. Note that in this case, the problem can only 
  be solved with continuous integration. Internally, `VariableRateJump` is 
  transformed into a `ContinuousCallback`. The `rate(u, p, t)` is used to 
  construct the `condition` function that triggers the callback.
  
#### Defining a Mass Action Jump

The constructor for a `MassActionJump` is:
```julia
MassActionJump(reactant_stoich, net_stoich; scale_rates = true, param_idxs=nothing)
```
- `reactant_stoich` is a vector whose `k`th entry is the reactant stoichiometry
  of the `k`th reaction. The reactant stoichiometry for an individual reaction
  is assumed to be represented as a vector of `Pair`s, mapping species id to
  stoichiometric coefficient.
- `net_stoich` is assumed to have the same type as `reactant_stoich`; a
  vector whose `k`th entry is the net stoichiometry of the `k`th reaction. The
  net stoichiometry for an individual reaction is again represented as a vector
  of `Pair`s, mapping species id to the net change in the species when the
  reaction occurs.
- `scale_rates` is an optional parameter that specifies whether the rate
  constants correspond to stochastic rate constants in the sense used by
  Gillespie, and hence need to be rescaled. *The default, `scale_rates=true`,
  corresponds to rescaling the passed in rate constants.* See below.
- `param_idxs` is a vector of the indices within the parameter vector, `p`, that
  correspond to the rate constant for each jump.

**Notes for Mass Action Jumps**
- When using `MassActionJump` the default behavior is to assume rate constants
  correspond to stochastic rate constants in the sense used by Gillespie (J.
  Comp. Phys., 1976, 22 (4)). This means that for a reaction such as ``2A
  \overset{k}{\rightarrow} B``, the jump rate function constructed by
  `MassActionJump` would be `k*A*(A-1)/2!`. For a trimolecular reaction like
  ``3A \overset{k}{\rightarrow} B`` the rate function would be
  `k*A*(A-1)*(A-2)/3!`. To *avoid* having the reaction rates rescaled (by `1/2`
  and `1/6` for these two examples), one can pass the `MassActionJump`
  constructor the optional named parameter `scale_rates=false`, i.e. use
  ```julia
  MassActionJump(reactant_stoich, net_stoich; scale_rates = false, param_idxs)
  ```
- Zero order reactions can be passed as `reactant_stoich`s in one of two ways.
  Consider the ``\varnothing \overset{k}{\rightarrow} A`` reaction with rate `k=1`:
  ```julia
  p = [1.]
  reactant_stoich = [[0 => 1]]
  net_stoich = [[1 => 1]]
  jump = MassActionJump(reactant_stoich, net_stoich; param_idxs=[1])
  ```
  Alternatively one can create an empty vector of pairs to represent the reaction:
  ```julia
  p = [1.]
  reactant_stoich = [Vector{Pair{Int,Int}}()]
  net_stoich = [[1 => 1]]
  jump = MassActionJump(reactant_stoich, net_stoich; param_idxs=[1])
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

#### Defining a Regular Jump

The constructor for a `RegularJump` is:

```julia
RegularJump(rate,c,numjumps;mark_dist = nothing)
```

- `rate(out,u,p,t)` is the function which computes the rate for every regular
  jump process
- `c(du,u,p,t,counts,mark)` calculates the update given `counts` number of
  jumps for each jump process in the interval.
- `numjumps` is the number of jump processes, i.e. the number of `rate` equations
  and the number of `counts`
- `mark_dist` is the distribution for the mark.

## Defining a Jump Problem

To define a `JumpProblem`, you must first define the basic problem. This can be
a `DiscreteProblem` if there is no differential equation, or an ODE/SDE/DDE/DAE
if you would like to augment a differential equation with jumps. Denote this
previously defined problem as `prob`. Then the constructor for the jump problem is:

```julia
JumpProblem(prob,aggregator::Direct,jumps::JumpSet;
            save_positions = typeof(prob) <: AbstractDiscreteProblem ? (false,true) : (true,true))
```

The aggregator is the method for simulating jumps. They are called aggregators
since they combine all `jumps` in a single discrete simulation algorithm.
Aggregators are defined below. `jumps` is a `JumpSet` which is just a gathering
of jumps. Instead of passing a `JumpSet`, one may just pass a list of jumps
themselves. For example:

```julia
JumpProblem(prob,aggregator,jump1,jump2)
```

and the internals will automatically build the `JumpSet`. `save_positions`
determines whether to save the state of the system  just before and/or after
events occur.

Note that a `JumpProblem`/`JumpSet` can only have 1 `RegularJump` (since a
`RegularJump` itself describes multiple processes together). Similarly, it can
only have one `MassActionJump` (since it also describes multiple processes
together).

## Jump Aggregators for Exact Simulation

Jump aggregators are methods for simulating jumps exactly. They are called
aggregators since they combine all `jumps` in a single discrete simulation
algorithm. Aggregators combine `jump` in different ways and offer different
trade-offs. However, all aggregators describe the realization of each and every
jump chronologically. Since they do not skip any jump, they are considered exact
methods. Note that none of the aggregators discussed in this section can be used
with `RegularJumps` which are used for inexact methods.

The current aggregators are:

- `Direct`: the Gillespie Direct method SSA.
- `RDirect`: A variant of Gillespie's Direct method that uses rejection to
  sample the next reaction.
- *`DirectCR`*: The Composition-Rejection Direct method of Slepoy et al. For
  large networks and linear chain-type networks it will often give better
  performance than `Direct`. (Requires dependency graph, see below.)
- `DirectFW`: the Gillespie Direct method SSA with `FunctionWrappers`. This
  aggregator uses a different internal storage format for collections of
  `ConstantRateJumps`.
- `FRM`: the Gillespie first reaction method SSA. `Direct` should generally
  offer better performance and be preferred to `FRM`.
- `FRMFW`: the Gillespie first reaction method SSA with `FunctionWrappers`.
- *`NRM`*: The Gibson-Bruck Next Reaction Method. For some reaction network
   structures this may offer better performance than `Direct` (for example,
   large, linear chains of reactions). (Requires dependency graph, see below.)
- *`RSSA`*: The Rejection SSA (RSSA) method of Thanh et al. With `RSSACR`, for
  very large reaction networks it often offers the best performance of all
  methods. (Requires dependency graph, see below.)
- *`RSSACR`*: The Rejection SSA (RSSA) with Composition-Rejection method of
  Thanh et al. With `RSSA`, for very large reaction networks it often offers the
  best performance of all methods. (Requires dependency graph, see below.)
- *`SortingDirect`*: The Sorting Direct Method of McCollum et al. It will
  usually offer performance as good as `Direct`, and for some systems can offer
  substantially better performance. (Requires dependency graph, see below.)
- *`QueueMethod`*: The queueing method. This is a modification of Ogata's
  algorihm for simulating any compound point process that evolves through time.
  This is the only aggregator that handles `VariableRateJump`. If rates do not
  change between jump events (i.e. `ConsantRateJump` or `MassActionJump`) this
  aggregator is very similar to `NRM`. (Requires dependency graph, see below.)

To pass the aggregator, pass the instantiation of the type. For example:

```julia
JumpProblem(prob,Direct(),jump1,jump2)
```

will build a problem where the constant rate jumps are solved using Gillespie's
Direct SSA method.

## Jump Aggregators Requiring Dependency Graphs
Italicized constant rate jump aggregators require the user to pass a dependency
graph to `JumpProblem`. `DirectCR`, `NRM`, `SortingDirect` and `QueueMethod`
require a jump-jump dependency graph, passed through the named parameter
`dep_graph`. i.e.
```julia
JumpProblem(prob,DirectCR(),jump1,jump2; dep_graph=your_dependency_graph)
```
For systems with only `MassActionJump`s, or those generated from a
[Catalyst](https://docs.sciml.ai/Catalyst/stable/) `reaction_network`, this graph
will be auto-generated. Otherwise, you must construct the dependency graph 
whenever using `ConstantRateJump`s and/or `VariableRateJump`s. This is also the 
case when combining `MassActionJump` with `ConstantRateJump`s and/or 
`VariableRateJump`s. 

Dependency graphs are represented as a `Vector{Vector{Int}}`, with the `i`th
vector containing the indices of the jumps for which rates must be recalculated
when the `i`th jump occurs. Internally, all `MassActionJump`s are ordered before
`ConstantRateJump`s and `VariableRateJump`s (with the latter internally ordered
in the same order they were passed in). Thus, keep that in mind when combining 
`MassActionJump`s with other types of jumps.

`RSSA` and `RSSACR` require two different types of dependency graphs, passed
through the following `JumpProblem` kwargs:
- `vartojumps_map` - A `Vector{Vector{Int}}` mapping each variable index, `i`,
  to a set of jump indices. The jump indices correspond to jumps with rate
  functions that depend on the value of `u[i]`.
-  `jumptovars_map` - A `Vector{Vector{Int}}`  mapping each jump index to a set
   of variable indices. The corresponding variables are those that have their
   value, `u[i]`, altered when the jump occurs.

For systems generated from a [Catalyst](https://docs.sciml.ai/Catalyst/stable/)
`reaction_network` these will be auto-generated. Otherwise you must explicitly
construct and pass in these mappings.

## Recommendations for exact methods
For representing and aggregating jumps
- Use a `MassActionJump` to handle all jumps that can be represented as mass
  action reactions with constant rate between jumps. This will generally offer 
  the fastest performance.
- Use `ConstantRateJump`s for any remaining jumps with constant rate between 
  jumps.
- Use `VariableRateJump`s for any remaining jumps with variable rate between 
  jumps. You will need to define the lower and upper rate boundaries as well as 
  the interval for which the boundaries apply. The tighter the boundaries and 
  the easier to compute, the faster the resulting algorithm will be. 
- For a small number of jumps, < ~10, `Direct` will often perform as well as the
  other aggregators.
- For > ~10 jumps `SortingDirect` will often offer better performance than `Direct`.
- For large numbers of jumps with sparse chain like structures and similar jump
  rates, for example continuous time random walks, `RSSACR`, `DirectCR` and then
  `NRM` often have the best performance.
- For very large networks, with many updates per jump, `RSSA` and `RSSACR` will
  often substantially outperform the other methods.
- For systems with `VariableRateJump`, only the `QueueMethod` aggregator is 
  supported.
- The `SSAStepper()` can be used with `VariableRateJump`s that modify the state
  of differential equations. However, it is not possible to use `SSAStepper()`
  to solve `VariableRateJump` that are combined with differential equations that
  modify the rate of the jumps.

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
rate1(u,p,t) = (0.1/1000.0)*u[1]*u[2]
function affect1!(integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump = ConstantRateJump(rate1,affect1!)

rate2(u,p,t) = 0.01u[2]
function affect2!(integrator)
  integrator.u[2] -= 1
  integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate2,affect2!)
u0    = [999,1,0]
p     = (0.1/1000,0.01)
tspan = (0.0,250.0)
dprob = DiscreteProblem(u0, tspan, p)
jprob = JumpProblem(dprob, Direct(), jump, jump2)
sol   = solve(jprob, SSAStepper())
```
We can change any of `u0`, `p` and `tspan` by either making a new
`DiscreteProblem`
```julia
u02    = [10,1,0]
p2     = (.1/1000, 0.0)
tspan2 = (0.0,2500.0)
dprob2 = DiscreteProblem(u02, tspan2, p2)
jprob2 = remake(jprob, prob=dprob2)
sol2   = solve(jprob2, SSAStepper())
```
or by directly remaking with the new parameters
```julia
jprob2 = remake(jprob, u0=u02, p=p2, tspan=tspan2)
sol2   = solve(jprob2, SSAStepper())
```
To avoid ambiguities, the following will give an error
```julia
jprob2 = remake(jprob, prob=dprob2, u0=u02)
```
as will trying to update either `p` or `tspan` while passing a new
`DiscreteProblem` using the `prob` kwarg.
