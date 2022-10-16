# [Piecewise Deterministic Markov Processes and Jump Diffusion Equations](@id jump_diffusion_tutorial)

!!! note

    This tutorial assumes you have read the [Ordinary Differential Equations tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/) in [`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable).

Jump Diffusion equations are stochastic differential equations with discontinuous
jumps. These can be written as:

```math
du = f(u,p,t)dt + \sum_{j}g_j(u,p,t)dW_j(t) + \sum_{i}h_i(u,p,t)dN_i(t)
```
where ``N_i`` is a Poisson-counter which denotes jumps of size ``h_i``. In this
tutorial we will show how to solve problems with even more general jumps. In the
special case that ``g_j = 0`` for all ``j``, we'll call the resulting jump-ODE a
[piecewise deterministic Markov
process](https://en.wikipedia.org/wiki/Piecewise-deterministic_Markov_process).

Before running this tutorial please install the following packages if they are
not already installed
```julia
using Pkg
Pkg.add("DifferentialEquations")
Pkg.add("Plots")
```
DifferentialEquations.jl will install JumpProcesses, along with the needed ODE and
SDE solvers.

We then load these packages, and set some plotting defaults, as
```@example tut3
using DifferentialEquations, Plots
default(; lw = 2)
```

## Defining a `ConstantRateJump` Problem
To start, let's solve an ODE that is coupled to a [`ConstantRateJump`](@ref). A
jump is defined as being "constant rate" if the rate is only dependent on values
from other `ConstantRateJump`s or [`MassActionJump`](@ref)s (a special type of
`ConstantRateJump`). This means that its rate must not be coupled with time, the
solution to the differential equation, or a solution component that is changed
by a [`VariableRateJump`](@ref). `ConstantRateJumps` are cheaper to compute than
`VariableRateJump`s, and so should be preferred when mathematically appropriate.

(Note: if your rate is only "slightly" dependent on the solution of the differential
equation, then it may be okay to use a `ConstantRateJump`. Accuracy loss will be
related to the percentage that the rate changes over the jump intervals.)

Let's solve the following problem. We will have a linear ODE with a Poisson counter
of rate 2 (which is the mean and variance), where at each jump the current solution
will be halved. To solve this problem, we first define the `ODEProblem`:
```@example tut3
function f(du,u,p,t)
    du[1] = u[1]
    nothing
end

prob = ODEProblem(f, [0.2], (0.0, 10.0))
```
Notice that, even though our equation is scalar, we define it using the in-place
array form. Variable rate jump equations will require this form. Note that for
this tutorial we solve a one-dimensional problem, but the same syntax applies
for solving a system of differential equations with multiple jumps.

Now we define our `rate` equation for our jump. Since it's just the constant
value 2, we do:
```@example tut3
rate(u, p, t) = 2
```
Now we define the `affect!` of the jump. This is the same as an `affect!` from a
[`DiscreteCallback`](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/),
and thus acts directly on the
[integrator](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/).
Therefore, to make it halve the current value of `u`, we do:
```@example tut3
function affect!(integrator)
    integrator.u[1] = integrator.u[1] / 2
    nothing
end
```
Then we build our jump:
```@example tut3
jump = ConstantRateJump(rate, affect!)
```
Next, we extend our `ODEProblem` to a [`JumpProblem`](@ref) by attaching the
jump:
```@example tut3
jump_prob = JumpProblem(prob, Direct(), jump)
```
We can now solve this extended problem using any ODE solver:
```@example tut3
sol = solve(jump_prob, Tsit5())
plot(sol)
```

## Variable Rate Jumps
Now let's define a jump with a rate that is dependent on the differential
equation via the solution vector. Let's set the rate to be the current value of
the solution, that is:
```@example tut3
rate(u,p,t) = u[1]
```
Using the same `affect!` we build a [`VariableRateJump`](@ref):
```@example tut3
jump = VariableRateJump(rate, affect!)
```
To make things interesting, let's copy this jump:
```@example tut3
jump2 = deepcopy(jump)
```
so that way we have two independent jump processes. We now couple these jumps
to the `ODEProblem`:
```@example tut3
jump_prob = JumpProblem(prob, Direct(), jump, jump2)
```
which we once again solve using an ODE solver:
```@example tut3
sol = solve(jump_prob, Tsit5())
plot(sol)
```
In this way we have solve a mixed jump-ODE, i.e. a piecewise deterministic
Markov process.

## Jump Diffusion
Now we will finally solve the jump diffusion problem. The steps are the same
as before, except we now start with a `SDEProblem` instead of an `ODEProblem`.
Using the same drift function `f` as before, we add multiplicative noise via:
```@example tut3
function g(du, u, p, t)
  du[1] = u[1]
  nothing
end

prob = SDEProblem(f, g, [0.2], (0.0, 10.0))
```
and couple it to the jumps:
```@example tut3
jump_prob = JumpProblem(prob, Direct(), jump, jump2)
```
We then solve it using an SDE algorithm:
```@example tut3
sol = solve(jump_prob, SRIW1())
plot(sol)
```
