# [Simple Poisson Processes in JumpProcesses] (@id poisson_proc_tutorial)

In this tutorial we show how to simulate several Poisson jump processes, for
several types of intensities and jump distributions. Readers interested
primarily in chemical or population process models, where several types of jumps
may occur, can skip directly to the [second tutorial](@ref ssa_tutorial) for a
tutorial covering similar material but focused on the SIR model.

JumpProcesses allows the simulation of jump processes where the transition rate,
i.e. intensity or propensity, can be a function of the current solution, current
parameters, and current time. Throughout this tutorial these are denoted by `u`,
`p` and `t`. Likewise, when a jump occurs any
DifferentialEquations.jl-compatible change to the current system state, as
encoded by a [DifferentialEquations.jl
integrator](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/), is
allowed. This includes changes to the current state or to parameter values (for
example via a callback).

This tutorial requires several packages, which can be added if not already
installed via
```julia
using Pkg
Pkg.add("JumpProcesses")
Pkg.add("Plots)
```
Let's also load our packages and set some defaults for our plot formatting
```@example tut1
using JumpProcesses, Plots
default(; lw = 2)
```

## `ConstantRateJump`s
Our first example will be to simulate a simple Poisson counting process,
``N(t)``, with a constant transition rate of λ. We can interpret this as a birth
process where new individuals are created at the constant rate λ. ``N(t)`` then
gives the current population size. In terms of a unit Poisson counting process,
``Y_b(t)``, we have
```math
N(t) = Y_b\left( \lambda t \right).
```
(Here by a unit Poisson counting process we just mean a Poisson counting process
with a constant rate of one.)

In the remainder of this tutorial we will use
*transition rate*, *rate*, *propensity*, and *intensity* interchangeably. Here
is the full program listing we will subsequently explain line by line
```julia
using JumpProcesses, Plots

rate(u,p,t) = p.λ
affect!(integrator) = (integrator.u[1] += 1)
crj = ConstantRateJump(rate, affect!)

u₀ = [0]
p = (λ = 2.0, )
tspan = (0.0, 10.0)

dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), crj)

sol = solve(jprob, SSAStepper())
plot(sol, label="N(t)", xlabel="t", legend=:bottomright)
```

We can define and simulate our jump process as follows. We first load our
packages
```@example tut1
using JumpProcesses, Plots
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
integrator](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/),
and directly modify the current solution value it stores. i.e. `integrator.u` is
the current solution vector, with `integrator.u[1]` the first component of this
vector. In our case we will only have one unknown, so this will be the current
value of the counting process. As our jump process's transition rate is constant
between jumps, we can use a [`ConstantRateJump`](@ref) to encode it
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
we are simulating a process that evolves in discrete time steps.
```@example tut1
dprob = DiscreteProblem(u₀, tspan, p)
```
We next create a [`JumpProblem`](@ref) that wraps the discrete problem, and
specifies which algorithm, called an aggregator in JumpProcesses, to use for
determining next jump times (and in the case of multiple possible jumps the next
jump type). Here we use the classical `Direct` method, proposed by Gillespie in
the chemical reaction context, but going back to earlier work by Doob and others
(and also known as Kinetic Monte Carlo in the physics literature)
```@example tut1
# a jump problem, specifying we will use the Direct method to sample
# jump times and events, and that our jump is encoded by crj
jprob = JumpProblem(dprob, Direct(), crj)
```
We are finally ready to simulate one realization of our jump process, selecting
`SSAStepper` to handle time-stepping our system from jump to jump
```@example tut1
# now we simulate the jump process in time, using the SSAStepper time-stepper
sol = solve(jprob, SSAStepper())

plot(sol, labels = "N(t)", xlabel = "t", legend = :bottomright)
```

### More general `ConstantRateJump`s
The previous counting process could be interpreted as a birth process, where new
individuals were created with a constant transition rate λ. Suppose we also
allow individuals to be killed with a death rate of μ. The transition rate at
time `t` for some individual to die, assuming the death of individuals are
independent, is just ``\mu N(t)``. Suppose we also wish to keep track of the
number of deaths, ``D(t)``, that have occurred. We can store these as an
auxiliary variable in `u[2]`. Our processes are then given mathematically by
```math
\begin{align*}
N(t) &= Y_b(\lambda t) - Y_d \left(\int_0^t \mu N(s^-) \, ds \right), \\
D(t) &= Y_d \left(\int_0^t \mu N(s^-) \, ds \right),
\end{align*}
```
where ``Y_d(t)`` denotes a second, independent, unit Poisson counting process.

We can encode this as a second jump for our system
like
```@example tut1
deathrate(u,p,t) = p.μ * u[1]
deathaffect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
deathcrj = ConstantRateJump(deathrate, deathaffect!)
```
As the death rate is constant *between* jumps we can encode this process as a
second `ConstantRateJump`. We then construct the corresponding problems, passing
both jumps to `JumpProblem`, and can solve as before
```@example tut1
p = (λ = 2.0, μ = 1.5)
u₀ = [0, 0]   # (N(0), D(0))
dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), crj, deathcrj)
sol = solve(jprob, SSAStepper())
plot(sol, labels = ["N(t)" "D(t)"], xlabel = "t", legend = :topleft)
```

In the next tutorial we will also introduce [`MassActionJump`](@ref)s, which are
a special type of [`ConstantRateJump`](@ref)s that require a more specialized
form of transition rate and state update, but can offer better computational
performance. They can encode any mass action reaction, as commonly arise in
chemical and population process models, and essentially require that
`rate(u,p,t)` be a monomial in the components of `u` and state changes be given
by adding or subtracting a constant vector from `u`.

## `VariableRateJump`s for processes that are not constant between jumps
So far we have assumed that our jump processes have transition rates that are
constant in between jumps. In many applications this may be a limiting
assumption. To support such models JumpProcesses has the
[`VariableRateJump`](@ref) type, which represents jump processes that have an
arbitrary time dependence in the calculation of the transition rate, including
transition rates that depend on states which can change in between two jumps
occurring. Let's consider the previous example, but now let the birth rate be
time dependent, ``b(t) = \lambda \left(\sin(\pi t / 2) + 1\right)``, so that our
model becomes
```math
\begin{align*}
N(t) &= Y_b\left(\int_0^t \left( \lambda \sin\left(\tfrac{\pi s}{2}\right) + 1 \right) \, d s\right) - Y_d \left(\int_0^t \mu N(s^-) \, ds \right), \\
D(t) &= Y_d \left(\int_0^t \mu N(s^-) \, ds \right).
\end{align*}
```


The birth rate is cyclical, bounded between a lower-bound of ``λ`` and an
upper-bound of ``2 λ``. We'll then re-encode the first (birth) jump as a
`VariableRateJump`. Two types of `VariableRateJump`s are supported, general and
bounded. The latter are generally more performant, but are also more restrictive
in when they can be used. They also require specifying additional information
beyond just `rate` and `affect!` functions.

Let's see how to build a bounded `VariableRateJump` encoding our new birth
process. We first specify the rate and affect functions, just like for a
`ConstantRateJump`,
```@example tut1
rate1(u,p,t) = p.λ * (sin(pi*t/2) + 1)
affect1!(integrator) = (integrator.u[1] += 1)
```
We next provide functions that determine a time interval over which the rate is
bounded from above given `u`, `p` and `t`. From these we can construct the new
bounded `VariableRateJump`:
```@example tut1
# We require that rate1(u,p,s) <= urate(u,p,s)
# for t <= s <= t + rateinterval(u,p,t)
rateinterval(u, p, t) = typemax(t)
urate(u, p, t) = 2 * p.λ

# Optionally, we can give a lower bound over the same interval.
# This may boost computational performance.
lrate(u, p, t) = p.λ

# now we construct the bounded VariableRateJump
vrj1 = VariableRateJump(rate1, affect1!; lrate, urate, rateinterval)
```

Finally, to efficiently simulate the new jump process we must also specify a
dependency graph. This indicates when a given jump occurs, which jumps in the
system need to have their rates and/or rate bounds recalculated (for example,
due to depending on changed components in `u`). We also assume the convention
that a given jump depends on itself. Since the first (birth) jump modifies the
population size `u[1]`, and the second (death) jump occurs at a rate
proportional to `u[1]`, when the first jump occurs we need to recalculate both
of the rates. In contrast, death does not change `u[1]`, and so the dependencies
of the second (death) jump are only itself. Note that the indices in the graph
correspond to the order in which the jumps appear when the problem is
constructed. The graph below encodes the dependents of the birth and death jumps
respectively
```@example tut1
dep_graph = [[1,2], [2]]
```

We can then construct the corresponding problem, passing both jumps to
`JumpProblem` as well as the dependency graph. We must use an aggregator that
supports bounded `VariableRateJump`s, in this case we choose the `Coevolve`
aggregator.
```@example tut1
jprob = JumpProblem(dprob, Coevolve(), vrj1, deathcrj; dep_graph)
sol = solve(jprob, SSAStepper())
plot(sol, labels = ["N(t)" "D(t)"], xlabel = "t", legend = :topleft)
```

If we did not know the upper rate bound or rate interval functions for the
time-dependent rate, we would have to use a continuous problem type and general
`VariableRateJump` to correctly handle calculating the jump times. Under this
assumption we would define a general `VariableRateJump` as following:
```@example tut1
vrj2 = VariableRateJump(rate1, affect1!)
```

Since the death rate now depends on a variable, `u[2]`, modified by a general
`VariableRateJump` (i.e. one that is not bounded), we also need to redefine the
death jump process as a general `VariableRateJump`
```@example tut1
deathvrj = VariableRateJump(deathrate, deathaffect!)
```

To simulate our jump process we now need to construct a continuous problem type
to couple the jumps to, for example an ordinary differential equation (ODE) or
stochastic differential equation (SDE). Let's use an ODE, encoded via an
`ODEProblem`. We simply set the ODE derivative to zero to preserve the state. We
are essentially defining a combined ODE-jump process, i.e. a [piecewise
deterministic Markov
process](https://en.wikipedia.org/wiki/Piecewise-deterministic_Markov_process),
but one where the ODE is trivial and does not change the state. To use this
problem type and the ODE solvers we first load `OrdinaryDiffEq.jl` or
`DifferentialEquations.jl`. If neither is installed, we first
```julia
using Pkg
Pkg.add("OrdinaryDiffEq")
# or Pkg.add("DifferentialEquations")
```
and then load it via
```@example tut1
using OrdinaryDiffEq
# or using DifferentialEquations
```
We can then construct our ODE problem with a trivial ODE derivative component.
Note, to work with the ODE solver time stepper we must also change our initial
condition to be floating point valued
```@example tut1
function f!(du, u, p, t)
    du .= 0
    nothing
end
u₀ = [0.0, 0.0]
oprob = ODEProblem(f!, u₀, tspan, p)
jprob = JumpProblem(oprob, Direct(), vrj2, deathvrj)
```
We can now simulate our jump process, using the `Tsit5` ODE solver as the time
stepper in place of `SSAStepper`
```@example tut1
sol = solve(jprob, Tsit5())
plot(sol, label=["N(t)" "D(t)"], xlabel="t", legend=:topleft)
```

For more details on when bounded vs. general `VariableRateJump`s can be used,
see the [next tutorial](@ref ssa_tutorial) and the [Jump Problems](@ref
jump_problem_type) documentation page.

## Having a Random Jump Distribution
Suppose we want to simulate a compound Poisson process, ``G(t)``, where
```math
G(t) = \sum_{i=1}^{N(t)} C_i
```
with ``N(t)`` a Poisson counting process with constant transition rate
``\lambda``, and the ``C_i`` independent and identical samples from a uniform
distribution over ``\{-1,1\}``. We can simulate such a process as follows.

We first ensure that we use the same random number generator as JumpProcesses. We
can either pass one as an input to [`JumpProblem`](@ref) via the `rng` keyword
argument, and make sure it is the same one we use in our `affect!` function, or
we can just use the default generator chosen by JumpProcesses if one is not
specified, `JumpProcesses.DEFAULT_RNG`. Let's do the latter
```@example tut1
rng = JumpProcesses.DEFAULT_RNG
```
Let's assume `u[1]` is ``N(t)`` and `u[2]` is ``G(t)``. We now proceed as in the
previous examples
```@example tut1
rate3(u,p,t) = p.λ

# define the affect function via a closure
affect3! = integrator -> let rng=rng
    # N(t) <-- N(t) + 1
    integrator.u[1] += 1

    # G(t) <-- G(t) + C_{N(t)}
    integrator.u[2] += rand(rng, (-1,1))
    nothing
end
crj = ConstantRateJump(rate3, affect3!)

u₀ = [0, 0]
p = (λ = 1.0,)
tspan = (0.0, 100.0)
dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), crj)
sol = solve(jprob, SSAStepper())
plot(sol, label=["N(t)" "G(t)"], xlabel="t")
```
