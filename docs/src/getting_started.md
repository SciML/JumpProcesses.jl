# Getting Started with JumpProcesses in Julia

Jumps (or point) processes are stochastic processes with discrete state changes
driven by a `rate` function. Historically, jump processes have been developed in
the context of dynamical systems to describe dynamics with sudden changes —
the jumps — in a system's value at random times. In contrast, the development
of point processes has been more focused on describing the occurrence of random
events — the points — over a support. In reality, jump and point processes
share many things in common which make JumpProcesses ideal for both.

Processes involving multiple jumps are known as compound jump (or point)
processes.

```math
du = \sum_{j} h_j(u,p,t) dN_j(t)
```

where ``N_j`` is a jump process with rate ``\lambda_i(u,p,t)``.

The homogeneous Poisson process is the canonical point process with a constant
rate of change. A compound Poisson process is a continuous-time Markov Chain
where the time to the next jump is exponentially distributed as determined by
the aggregate rate. In the statistics literature, the composition of Poisson
processes is described by the superposition theorem. Simulation algorithms for
these types of processes are known in biology and chemistry as Gillespie methods
or Stochastic Simulation Algorithms (SSA), with the time evolution that the
probability these processes are in a given state at a given time satisfying the
Chemical Master Equation (CME).

Any differential equation can be extended by jumps. For example, we have an ODE
with jumps, denoted by

```math
du = f(u,p,t)dt + \sum_{j} h_j(u,p,t) dN_j(t)
```

Extending a stochastic differential equation (SDE) to have jumps is commonly known as a jump-
diffusion, and is denoted by

```math
du = f(u,p,t)dt + \sum_{i}g_i(u,t)dW_i(t) + \sum_{j}h_i(u,p,t)dN_i(t)
```

The general workflow with any of the jump processes above is to define the base
and jump problem, solve the jump problem and then analyze the solution. The full
code for a jump process with no other dynamics apart from jumps is:

```@example ex0
using JumpProcesses
u0 = [0]
tspan = (0.0, 10.0)
dprob = DiscreteProblem(u0, tspan)
rate(u, p, t) = 2.0
affect!(integrator) = (integrator.u[1] += 1)
jump = ConstantRateJump(rate, affect!)
jprob = JumpProblem(dprob, Direct(), jump)
sol = solve(jprob, SSAStepper())

using Plots
plot(sol, title = "Sample path from a jump process with constant rate",
    label = "N(t)", xlabel = "t", legend = :bottomright)
```

## Step 1: Defining a problem

The first thing you want to do is to define your base problem from the many
options available. For dynamics that involve only jumps we employ
a `DiscreteProblem` as our base problem.

```@example ex0
using JumpProcesses
u0 = [0]
tspan = (0.0, 10.0)
dprob = DiscreteProblem(u0, tspan)
```

For our example, notice that `u0` is a `Int[]` which is an appropriate choice in
this case because we will only be working with jumps. Since we will be modifying
`u0` only when a jump occurs via its `affect!`, we initialize `DiscreteProblem`
without any function mapping which means `DiscreteProblem` use the default
identity mapping.

JumpProcesses exports
[`DiscreteProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/)
from OrdinaryDiffEq. In case you want
to model the base dynamics as an
[`ODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/) or as
an [`SDEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/sde_types/)
you will need to import
OrdinaryDiffEq.

## Step 2: Defining a jump

Jumps are implemented as [callbacks
functions](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/)
from DifferentialEquations. Our library takes care of adding the appropriate
callbacks to the base problem to ensure that jumps are properly initialized and
executed.

Here we add a `ConstantRateJump` to the base problem. As such the user must
define the `rate` and `affect!` which determines the frequency and the effect of
the jumps on the base problem.

```@example ex0
rate(u, p, t) = 2.0
affect!(integrator) = (integrator.u[1] += 1)
jump = ConstantRateJump(rate, affect!)
```

There are many types of jumps. Some of them allow for variable rates. You can
read about [different jump types in this page](@ref jump_types).

Once the jump is defined, we initialize the `JumpProblem`. _Aggegators_ are
algorithms that determines jump times. We call them _aggregators_ because they
aggregate all jump callbacks into a single callback. Alternatively, we can think
of aggregators as the jump simulation algorithms. In this case we use the
`Direct` _aggregator_.

```@example ex0
jprob = JumpProblem(dprob, Direct(), jump)
```

JumpProblem can be initialized with different combination of jumps and
aggregators. If you want to understand more about the options available read
about [JumpProblem initialization in here](@ref defining_jump_problem).

In addition to that, aggregators come with different trade-offs. Not all
aggregators accept all jump types. To learn [more about _aggregators_ check this
section](@ref Jump-Aggregators-for-Exact-Simulation).

## Step 3: Solving a problem

After defining a problem, we solve it using `solve` with an appropriate stepper.

```@example ex0
sol = solve(jprob, SSAStepper())
```

Steppers tell how we evolve time in our base problem and required for any
numerical simulator. JumpProcesses offers the `SSAStepper` which steps through
time one jump candidate at a time.

If you are modelling other types of base problems like `ODEProblem` or
`SDEProblem`, you will not be able to use `SSAStepper` since these problems
require more fine-grained time evolution.

Apart from time-stepping, you might also be interested in controlling the saving
frequency of the state variable `u`. This control can be thought as orthogonal
to how the stepper evolves time. To avoid saving at every jump, we can
initialize jumps as following.

```@example ex0
jprob = JumpProblem(dprob, Direct(), jump; save_positions = (false, false))
```

Finally, to solve the problem at regular intervals we can use `saveat`.

```@example ex0
sol = solve(jprob, SSAStepper(); saveat = 1.0)
```

When you do not save the jump events, be careful when analysing interpolated
values as they will not be an accurate representation of the sampled path. This
can be particularly problematic plotting the data.

## Going Beyond the Poisson process: How to Use the Documentation

This tutorial covered only the basics of the Poisson process with constant rate.
We mostly focused on the basic pattern to define and solve a JumpProblem.

In case you want to go further, JumpProcesses is a component package in the
[SciML](https://sciml.ai/) ecosystem, and one of the core solver libraries
included in
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/).

The documentation includes tutorials and examples:

  - [Simulating basic Poisson processes](@ref poisson_proc_tutorial)
  - [Simulating jump processes via SSAs (i.e., Gillespie methods)](@ref ssa_tutorial)
  - [Simulating jump-diffusion processes](@ref jump_diffusion_tutorial)
  - [Temporal point processes (TPP)](@ref tpp_tutorial)
  - [Spatial SSAs](@ref Spatial-SSAs-with-JumpProcesses.jl)

In addition to that the document contains references to guide you through:

  - [References on the types of jumps and available simulation methods](@ref jump_problem_type)
  - [References on jump time stepping methods](@ref jump_solve)
  - [FAQ with information on changing parameters between simulations and using callbacks](@ref FAQ)
  - [API documentation](@ref JumpProcesses.jl-API)
