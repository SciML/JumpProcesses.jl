# [Simple Poisson Processes in DiffEqJump] (@id poisson_proc_tutorial)

In this tutorial we show how to simulate several Poisson jump processes, for
several types of intensities and jump distributions. Readers interested
primarily in chemical or population process models, where several types of jumps
may occur, can skip directly to the [second tutorial](@ref ssa_tutorial) for a
tutorial covering similar material but focused on the SIR model.

DiffEqJump allows the simulation of jump processes where the transition rate, i.e.
intensity or propensity, can be a function of the current solution, current
parameters, and current time. Throughout this tutorial these are denoted by `u`,
`p` and `t`. Likewise, when a jump occurs any
DifferentialEquations.jl-compatible change to the current system state, as
encoded by a [DifferentialEquations.jl
integrator](https://docs.sciml.ai/dev/modules/DiffEqDocs/basics/integrator/), is
allowed. This includes changes to the current state or to parameter values.

This tutorial requires several packages, which can be added if not already
installed via
```julia
using Pkg
Pkg.add("DiffEqJump")
Pkg.add("Plots)
```

## `ConstantRateJump`s
Our first example will be to simulate a simple Poission counting process with a
constant transition rate of `2`. In the remainder of this tutorial we will use
*transition rate*, *rate*, *propensity*, and *intensity* interchangeably. Here
is the full program listing we will subsequently explain line by line
```julia
using DiffEqJump, Plots

rate(u,p,t) = p.λ
affect!(integrator) = (integrator.u[1] += 1)
crj = ConstantRateJump(rate, affect!)

u₀ = [0]
p = (λ = 2.0, )
tspan = (0.0, 10.0)

dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), crj)

sol = solve(jprob, SSAStepper())
plot(sol, label="Number of jumps by time t", legend=:bottomright)
```

We can define and simulate our jump process using DiffEqJump. We first load our
packages
```@example tut1
using DiffEqJump, Plots
```
To specify our jump process we need to define two functions. One that given the
current state of the system, `u`, the parameters, `p`, and the time, `t`, can
determine the current transition rate (intensity)
```@example tut1
rate(u,p,t) = p.λ
```
This corresponds to the instantaneous probability per time a jump occurs when
the current state is `u`, current parameters are `p`, and the time is `t`. We
also give a function that updates the system state when a jump is known to have
occurred (at time `integrator.t`)
```@example tut1
affect!(integrator) = (integrator.u[1] += 1)
```
Here the convention is to take a [DifferentialEquations.jl
integrator](https://docs.sciml.ai/dev/modules/DiffEqDocs/basics/integrator/),
and directly modify the current solution value it stores. i.e. `integrator.u` is
the current solution vector, with `integrator.u[1]` the first component of this
vector. In our case we will only have one unknown, so this will be the current
value of the counting process. As our jump process' transition rate is constant
between jumps, we can use a `ConstantRateJump` to encode it
```@example tut1
crj = ConstantRateJump(rate, affect!)
```

We then specify the parameters needed to simulate our jump process
```@example tut1
# the initial condition vector, notice we make it an integer
# since we have a discrete counting process
u₀ = [0]

# the parameters of the model, in this case a named tuple storing the rate, λ
p = (λ = 2.0, )

# the time interval to solve over
tspan = (0.0, 10.0)
```
Finally, we construct the associated SciML problem types and generate one
realization of the process. We first create a `DiscreteProblem` to encode that
we are simulating a process that evolves in discrete time steps. Note, this
currently requires that the process has constant transition rates *between*
jumps
```@example tut1
dprob = DiscreteProblem(u₀, tspan, p)
```
We next create a `JumpProblem` that wraps the discrete problem, and specifies
which algorithm to use for determining next jump times (and in the case of
multiple possible jumps the next jump type). Here we use the classical `Direct`
method, proposed by Gillespie in the chemical reaction context, but going back
to earlier work by Doob and others (and also known as Kinetic Monte Carlo in the
physics literature)
```@example tut1
# a jump problem, specifying we will use the Direct method to sample
# jump times and events, and that our jump is encoded by crj
jprob = JumpProblem(dprob, Direct(), crj)
```
We are finally ready to simulate one realization of our jump process. Here we
```@example tut1
# now we simulate the jump process in time, using the SSAStepper time-stepper
sol = solve(jprob, SSAStepper())

plot(sol, label="Number of jumps by time t", legend=:bottomright)
```