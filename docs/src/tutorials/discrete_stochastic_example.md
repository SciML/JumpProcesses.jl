# [Continuous-Time Jump Processes and Gillespie Methods](@id ssa_tutorial)

In this tutorial, we will describe how to define and simulate continuous-time
jump (or point) processes, also known in biological fields as stochastic
chemical kinetics (i.e., Gillespie) models. It is not necessary to have read the
[first tutorial](@ref poisson_proc_tutorial). We will illustrate:

  - The different types of jumps that can be represented in JumpProcesses and their
    use cases.
  - How to speed up pure-jump simulations with only [`ConstantRateJump`](@ref)s,
    [`MassActionJump`](@ref)s, and bounded `VariableRateJump`s by using the
    [`SSAStepper`](@ref) time stepper.
  - How to define and use [`MassActionJump`](@ref)s, a more specialized type of
    [`ConstantRateJump`](@ref) that offers improved computational performance.
  - How to define and use bounded [`VariableRateJump`](@ref)s in pure-jump simulations.
  - How to use saving controls to reduce memory use per simulation.
  - How to use general [`VariableRateJump`](@ref)s and when they should be
    preferred over the other jump types.
  - How to create hybrid problems, mixing the various jump types with ODEs or SDEs.
  - How to use `RegularJump`s to enable faster, but approximate, time stepping via
    τ-leaping methods.

!!! note
    
    This tutorial assumes you have read the [Ordinary Differential Equations tutorial](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/) in [`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable).

We begin by demonstrating how to build jump processes using
[JumpProcesses.jl](https://docs.sciml.ai/JumpProcesses/stable/)'s different jump types,
which encode the rate functions (i.e., transition rates, intensities, or
propensities) and state changes when a given jump occurs.

Note, the SIR model considered here is a type of stochastic chemical kinetics
jump process model, and as such the biological modeling functionality of
[Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) can be used to easily
specify the model and automatically calculate inputs needed for JumpProcesses's
optimized simulation algorithms. We summarize this alternative approach at the
beginning for users who may be interested in modeling chemical systems, but note
this tutorial is intended to explain the general jump process formulation of
JumpProcesses for all users. However, for those users constructing models that can
be represented as a collection of chemical reactions we strongly recommend using
Catalyst, which should ensure optimal jump types are selected to represent each
reaction, and necessary data structures for the simulation algorithms, such as
dependency graphs, are automatically calculated.

We'll make use of the
[DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
meta package, which includes JumpProcesses and ODE/SDE solvers, Plots.jl, and
(optionally) Catalyst.jl in this tutorial. If not already installed, they can be
added as follows:

```julia
using Pkg
Pkg.add("DifferentialEquations")
Pkg.add("Plots")
Pkg.add("Catalyst")                # optional
```

Let's now load the required packages and set some default plot settings

```julia
using DifferentialEquations, Plots, LinearAlgebra
default(; lw = 2)
```

```@setup tut2
using DifferentialEquations, Plots, LinearAlgebra
default(; lw = 2)
```

## Illustrative Model: SIR Disease Dynamics

To illustrate the jump process solvers, we will build an SIR model which matches
the tutorial from [Gillespie.jl](https://github.com/sdwfrost/Gillespie.jl). SIR
stands for susceptible, infected, and recovered, and is a model of disease
spread. When a susceptible person comes in contact with an infected person, the
disease has a chance of infecting the susceptible person. This "chance" is
determined by the number of susceptible persons and the number of infected
persons, since in larger populations there is a greater chance that two people
come into contact. Every infected person will in turn have a rate at which they
recover. In our model, we'll assume there are no births or deaths, and a
recovered individual is protected from reinfection.

We'll begin by giving the mathematical equations for the jump processes of the
number of susceptible (``S(t)``), number of infected (``I(t)``), and number of
recovered (``R(t)``). In the next section, we give a more intuitive and
biological description of the model for users that are less familiar with jump
processes. Let ``Y_i(t)``, ``i = 1,2``, denote independent unit Poisson
processes. Our basic mathematical model for the evolution of
``(S(t),I(t),R(t))``, written using Kurtz's time-change representation, is then

```math
\begin{aligned}
S(t) &= S(0) - Y_1\left(  \int_0^t \beta S(s^{-}) I(s^{-}) \, ds\right) \\
I(t) &= I(0) + Y_1\left(  \int_0^t \beta S(s^{-}) I(s^{-}) \, ds\right)
        - Y_2 \left( \int_0^t \nu I(s^-)  \, ds \right) \\
R(t) &= R(0) + Y_2 \left( \int_0^t \nu I(s^-)  \, ds \right)
\end{aligned}
```

Notice, our model involves two jump processes with rate functions, also known as
intensities or propensities, given by ``\beta S(t) I(t)`` and ``\nu I(t)``
respectively. These give the probability per time a new infected individual is
created, and the probability per time some infected individual recovers.

For those less-familiar with the time-change representation, we next give a more
intuitive explanation of the model as a collection of chemical reactions, and
then demonstrate how these reactions can be written in
[Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) and seamlessly converted
into a form that can be used with the
[JumpProcesses.jl](https://docs.sciml.ai/JumpProcesses/stable/) solvers. Users
interested in how to directly define jumps using the lower-level JumpProcesses
interface can skip to [Building and Simulating the Jump Process Using the
JumpProcesses Low-Level Interface](@ref).

## Specifying the SIR Model with Chemical Reactions

The SIR model described above involves two basic chemical reactions,

```math
\begin{aligned}
S + I &\overset{\beta}{\to} 2 I \\
I &\overset{\nu}{\to} R,
\end{aligned}
```

where ``\beta`` and ``\nu`` are the rate constants of the reactions (with units
of probability per time). In a jump process (stochastic chemical kinetics)
model, we keep track of the non-negative integer number of each species at each
time (i.e., ``(S(t), I(t), R(t))`` above). Each reaction has an associated rate
function (i.e., intensity or propensity) giving the probability per time it can
occur when the system is in state ``(S(t),I(t),R(t))``:

```math
\begin{matrix}
\text{Reaction} & \text{Rate Functions} \\
\hline
S + I \overset{\beta}{\to} 2 I & \beta S(t) I(t) \\
I \overset{\nu}{\to} R & \nu I(t).
\end{matrix}
```

``\beta`` is determined by factors like the type of the disease. It can be
interpreted as the probability per time one pair of susceptible and infected
people encounter each other, with the susceptible person becoming sick. The
overall rate (i.e., probability per time) that some susceptible person gets sick
is then given by the rate constant multiplied by the number of possible pairs of
susceptible and infected people. This formulation is known as the [law of mass
action](https://en.wikipedia.org/wiki/Law_of_mass_action). Similarly, we have
that each individual infected person is assumed to recover with probability per
time ``\nu``, so that the probability per time *some* infected person recovers
is ``\nu`` times the number of infected people, i.e., ``\nu I(t)``.

Rate functions give the probability per time for each of the two types of jumps
to occur, and hence determine when the state of our system changes. To fully
specify our model, we also need to specify how the state changes when a jump
occurs, giving what are called `affect!` functions in JumpProcesses. For example,
when the $S + I \to 2 I$ reaction occurs, and some susceptible person becomes
infected, the subsequent (instantaneous) state change is that

```math
\begin{aligned}
S &\to S - 1 & I &\to I + 1.
\end{aligned}
```

Likewise, when the $I \to R$ reaction occurs so that some infected person becomes
recovered, the state change is

```math
\begin{aligned}
I &\to I - 1 & R \to R + 1.
\end{aligned}
```

In summary, our model is described by two chemical reactions, which each in turn
correspond to a jump process determined by a `rate` function specifying how
frequently jumps should occur, and an `affect!` function for how the state
should change when that jump type occurs.

## Building and Simulating the Jump Processes from Catalyst Models

Using [Catalyst.jl](https://docs.sciml.ai/Catalyst/stable/) we can input our full
reaction network in a form that can be easily used with JumpProcesses's solvers

```@example tut2
using Catalyst
sir_model = @reaction_network begin
    β, S + I --> 2I
    ν, I --> R
end
```

To build a pure jump process model of the reaction system, where the state is
constant between jumps, we will use a
[`DiscreteProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/).
This encodes that the state only changes at the jump times. We do this by giving
the constructor `u₀`, the initial condition, and `tspan`, the timespan. Here, we
will start with `999` susceptible people, `1` infected person, and `0` recovered
people, and solve the problem from `t=0.0` to `t=250.0`. We use the parameters
`β = 0.1/1000` and `ν = 0.01`. Thus, we build the problem via:

```@example tut2
p = (:β => 0.1 / 1000, :ν => 0.01)
u₀ = [:S => 999, :I => 10, :R => 0]
tspan = (0.0, 250.0)
prob = DiscreteProblem(sir_model, u₀, tspan, p)
```

*Notice, the initial populations are integers, since we want the exact number of
people in the different states.*

The Catalyst reaction network can be converted into various
DifferentialEquations.jl problem types, including `JumpProblem`s, `ODEProblem`s,
or `SDEProblem`s. To turn it into a [`JumpProblem`](@ref) representing the SIR jump
process model, we simply write

```@example tut2
jump_prob = JumpProblem(sir_model, prob, Direct())
```

Here `Direct()` indicates that we will determine the random times and types of
reactions using [Gillespie's Direct stochastic simulation algorithm
(SSA)](https://doi.org/10.1016/0021-9991(76)90041-3), also known as Doob's
method or Kinetic Monte Carlo. See [Jump Aggregators for Exact Simulation](@ref) for
other supported SSAs.

We now have a problem that can be evolved in time using the JumpProcesses solvers.
Since our model is a pure jump process (no continuously-varying components), we
will use `SSAStepper()` to handle time-stepping the `Direct` method from jump to
jump:

```@example tut2
sol = solve(jump_prob, SSAStepper())
```

This solve command takes the standard commands of the common interface, and the
solution object acts just like any other differential equation solution. Thus,
there exists a plot recipe, which we can plot with:

```@example tut2
plot(sol)
```

## Building and Simulating the Jump Process Using the JumpProcesses Low-Level Interface

We now show how to directly use JumpProcesses's low-level interface to construct
and solve our jump process model for ``(S(t),I(t),R(t))``. Each individual jump
that can occur is represented through specifying two pieces of information; a
`rate` function (i.e., intensity or propensity) for the jump and an `affect!`
function for the jump. The former gives the probability per time a particular
jump can occur given the current state of the system, and hence determines the
time at which jumps can happen. The latter specifies the instantaneous change in
the state of the system when the jump occurs.

In our SIR model, we have two possible jumps that can occur (one for susceptibles
becoming infected and one for infected becoming recovered), with the
corresponding (mathematical) rates and affects given by

```math
\begin{matrix}
\text{Rates} & \text{Affects}\\
\hline
\beta S(t) I(t) & S \to S - 1,\, I \to I + 1 \\
\nu I(t) & I \to I - 1, \, R \to R + 1.
\end{matrix}
```

JumpProcesses offers three different ways to (exactly) represent jumps:
`MassActionJump`, `ConstantRateJump`, and `VariableRateJump`. Choosing which to
use is a trade-off between the desired generality of the `rate` and `affect!`
functions vs. the computational performance of the resulting simulated system.
In general

| Jump Type                                                            | Performance                             | Generality                                                                                                                                 |
|:--------------------------------------------------------------------:|:---------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| [`MassActionJump`](@ref MassActionJumpSect)                          | Fastest                                 | Restrictive rates/affects                                                                                                                  |
| [`ConstantRateJump`](@ref ConstantRateJumpSect)                      | Somewhat Slower than `MassActionJump`   | Rate function must be constant between jumps                                                                                               |
| [`VariableRateJump` with rate bounds](@ref VariableRateJumpWithBnds) | Somewhat Slower than `ConstantRateJump` | Rate functions can explicitly depend on time, but require an upper bound that is guaranteed constant between jumps over some time interval |
| [`VariableRateJump`](@ref VariableRateJumpSect)                      | Slowest                                 | Completely general                                                                                                                         |

It is recommended to try to encode jumps using the most performant option that
supports the desired generality of the underlying `rate` and `affect!`
functions. Below we describe the different jump types, and show how the SIR
model can be formulated using first `ConstantRateJump`s, then more performant
`MassActionJump`s, and finally with `VariableRateJump`s using rate bounds. We
conclude by presenting several completely general models that use
`VariableRateJump`s without rate bounds, and which require an ODE solver to
handle time-stepping.

We note, in the remainder we will refer to *bounded* `VariableRateJump`s as
those for which we can specify functions calculating a time window over which
the rate is bounded by a constant (as long as the state is unchanged), see [the
section on bounded `VariableRateJump`s for details](@ref VariableRateJumpWithBnds).
`VariableRateJump`s or *general* `VariableRateJump`s will refer to those for
which such functions are not available.

## [Defining the Jumps Directly: `ConstantRateJump`](@id ConstantRateJumpSect)

The constructor for a `ConstantRateJump` is:

```julia
jump = ConstantRateJump(rate, affect!)
```

where `rate` is a function `rate(u,p,t)` and `affect!` is a function of the
integrator `affect!(integrator)` (for details on the integrator, see the
[integrator interface
docs](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/)). Here
`u` corresponds to the current state vector of the system; for our SIR model
`u[1] = S(t)`, `u[2] = I(t)` and `u[3] = R(t)`. `p` corresponds to the parameters of
the model, just as used for passing parameters to derivative functions in ODE
solvers. Thus, to define the two possible jumps for our model, we take (with
`β = .1/1000.0` and `ν = .01`).

```@example tut2
β = 0.1 / 1000.0
ν = 0.01;
p = (β, ν)
rate1(u, p, t) = p[1] * u[1] * u[2]  # β*S*I
function affect1!(integrator)
    integrator.u[1] -= 1         # S -> S - 1
    integrator.u[2] += 1         # I -> I + 1
    nothing
end
jump = ConstantRateJump(rate1, affect1!)

rate2(u, p, t) = p[2] * u[2]         # ν*I
function affect2!(integrator)
    integrator.u[2] -= 1        # I -> I - 1
    integrator.u[3] += 1        # R -> R + 1
    nothing
end
jump2 = ConstantRateJump(rate2, affect2!)
```

We will start with `999` susceptible people, `1` infected person, and `0`
recovered people, and solve the problem from `t=0.0` to `t=250.0` so that

```@example tut2
u₀ = [999, 10, 0]
tspan = (0.0, 250.0)
```

*Notice, the initial populations are integers, since we want the exact number of
people in the different states.*

Since we want the system state to change only at the discrete jump times, we
will build a
[`DiscreteProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/)

```@example tut2
prob = DiscreteProblem(u₀, tspan, p)
```

We can then use [`JumpProblem`](@ref) from JumpProcesses to augment the discrete
problem with jumps and select the stochastic simulation algorithm (SSA) to use
in sampling the jump processes. To create a `JumpProblem` we would simply do:

```@example tut2
jump_prob = JumpProblem(prob, Direct(), jump, jump2)
```

Here [`Direct()`](@ref) indicates that we will determine the random times and
types of jumps that occur using [Gillespie's Direct stochastic simulation
algorithm (SSA)](https://doi.org/10.1016/0021-9991(76)90041-3), also known as
Doob's method or Kinetic Monte Carlo. See [Jump Aggregators for Exact Simulation](@ref)
for other supported SSAs.

We now have a problem that can be evolved in time using the JumpProcesses solvers.
Since our model is a pure jump process with all rates being constant in between
jumps (i.e., no continuously-varying components), we will use
[`SSAStepper`](@ref) to handle time-stepping the `Direct` method from jump to
jump:

```@example tut2
sol = solve(jump_prob, SSAStepper())
```

This solve command takes the standard commands of the common interface, and the
solution object acts just like any other differential equation solution. Thus
there exists a plot recipe, which we can plot with:

```@example tut2
plot(sol, label = ["S(t)" "I(t)" "R(t)"])
```

Note, in systems with more than a few jumps (more than ~10), it can be
advantageous to use more sophisticated SSAs than `Direct`. For such systems it
is recommended to use [`SortingDirect`](@ref), [`RSSA`](@ref) or
[`RSSACR`](@ref), see the list of JumpProcesses SSAs at [Jump Aggregators for
Exact Simulation](@ref).

## SSAStepper

Any common interface algorithm can be used to perform the time-stepping, since it
is implemented over the callback interface. This allows for hybrid systems that
mix ODEs, SDEs and jumps. In many cases, we may have a pure jump system that only
involves `ConstantRateJump`s, `MassActionJump`s, and/or bounded
`VariableRateJump`s (see below). In those cases, a substantial performance
benefit may be gained by using [`SSAStepper`](@ref). Note, `SSAStepper` is a
more limited time-stepper which only supports discrete events, and does not
allow simultaneous coupled ODEs/SDEs, or general `VariableRateJump`s. It is,
however, very efficient for pure jump problems involving only
`ConstantRateJump`s, `MassActionJump`s, and bounded `VariableRateJump`s.

## [Defining the Jumps Directly: `MassActionJump`](@id MassActionJumpSect)

For `ConstantRateJump`s that can be represented as mass action reactions a
further specialization of the jump type is possible that offers improved
computational performance; [`MassActionJump`](@ref). Suppose the system has
``N`` chemical species ``\{S_1,\dots,S_N\}``. A general mass action reaction has
the form

```math
R_1 S_1 + R_2 S_2 + \dots + R_N S_N \overset{k}{\rightarrow} P_1 S_1 + P_2 S_2 + \dots + P_N S_N
```

where the non-negative integers ``(R_1,\dots,R_N)`` denote the *reactant
stoichiometry* of the reaction, and the non-negative integers
``(P_1,\dots,P_N)`` the *product stoichiometry*. The *net stoichiometry* is the
net change in each chemical species from the reaction occurring one time, given
by ``\mathbf{\nu} = (P_1-R_1,\dots,P_N-R_N)``. As such, the `affect!` function associated with
a `MassActionJump` simply changes the state, ``\mathbf{u}(t) = (u_1(t),\dots,u_N(t))``,
by updating

```math
\mathbf{u}(t) \to \mathbf{u}(t) + \mathbf{\nu}.
```

The default rate function, `ρ = rate(u,p,t)`, is based on stochastic chemical
kinetics and given by

```math
ρ(\mathbf{u}(t)) = k \prod_{i=1}^N \begin{pmatrix} u_i \\ R_i \end{pmatrix}
= k \prod_{i=1}^N \frac{u_i!}{R_i! (u_i - R_i)!}
= k \prod_{i=1}^N \frac{u_i (u_i - 1) \cdots (u_i - R_i + 1)}{R_i!}
```

where ``k`` denotes the rate constant of the reaction (in units of per time).

As an example, consider again the SIR model. The species are `u = (S,I,R)`. The
first reaction has rate `β`, reactant stoichiometry `(1, 1, 0)`, product
stoichiometry `(0, 2, 0)`, and net stoichiometry `(-1, 1, 0)`. The second reaction
has rate `ν`, reactant stoichiometry `(0, 1, 0)`, product stoichiometry `(0, 0, 1)`,
and net stoichiometry `(0, -1, 1)`.

We can manually encode this system as a mass action jump by specifying the
indexes of the rate constants in `p`, the reactant stoichiometry, and the net
stoichiometry. We note that the first two determine the rate function, with the
latter determining the affect function.

```@example tut2
rateidxs = [1, 2]           # i.e., [β, ν]
reactant_stoich = [
    [1 => 1, 2 => 1],         # 1*S and 1*I
    [2 => 1],                  # 1*I
]
net_stoich = [
    [1 => -1, 2 => 1],        # -1*S and 1*I
    [2 => -1, 3 => 1],         # -1*I and 1*R
]
mass_act_jump = MassActionJump(reactant_stoich, net_stoich; param_idxs = rateidxs)
```

Notice, one typically should define *one* `MassActionJump` that encodes each
possible jump that can be represented via a mass action reaction.
This contrasts with `ConstantRateJump`s or `VariableRateJump`s where separate instances
are created for each distinct jump type.

Just like for `ConstantRateJumps`, to then simulate the system we create
a `JumpProblem` and call `solve`:

```@example tut2
jump_prob = JumpProblem(prob, Direct(), mass_act_jump)
sol = solve(jump_prob, SSAStepper())
plot(sol; label = ["S(t)" "I(t)" "R(t)"])
```

For more details about MassActionJumps see [Defining a Mass Action Jump](@ref).
We note that one could include the factors of ``1 / R_i!`` directly in the rate
constant passed into a [`MassActionJump`](@ref), so that the desired rate
function that gets evaluated is

```math
\hat{k} \prod_{i=1}^N u_i (u_i - 1) \cdots (u_i - R_i + 1)
```

with ``\hat{k} = k / \prod_{i=1}^{N} R_i!`` the renormalized rate constant.
Passing the keyword argument `scale_rates = false` will disable
`MassActionJump`s internally rescaling the rate constant by ``(\prod_{i=1}^{N} R_i!)^{-1}``.

For chemical reaction systems Catalyst.jl automatically groups reactions
into their optimal jump representation.

### *Caution about ConstantRateJumps and MassActionJumps*

`ConstantRateJump`s and `MassActionJump`s are restricted in that they assume the
rate functions are constant at all times between two consecutive jumps of the
system. That is, any species/states or parameters that a rate function depends
on must not change between the times at which two consecutive jumps occur. Such
conditions are violated if one has a time-dependent parameter like ``\beta(t)``
or if some solution components, say `u[2]`, may also evolve through a
coupled ODE, SDE, or a general [`VariableRateJump`](@ref) (see below for
examples). For problems where the rate function may change between consecutive
jumps, bounded or general [`VariableRateJump`](@ref)s must be used.

Thus, in the examples above,

```julia
rate1(u, p, t) = p[1] * u[1] * u[2]
rate2(u, p, t) = p[2] * u[2]
```

both must be constant, besides changes due to some other `ConstantRateJump` or
`MassActionJump` (the same restriction applies to `MassActionJump`s). Since
these rates only change when `u[1]` or `u[2]` is changed, and `u[1]` and `u[2]`
only change when one of the jumps occur, this setup is valid. However, a rate of
`t*p[1]*u[1]*u[2]` would not be valid because the rate would change in between
jumps, as would `p[2]*u[1]*u[4]` when `u[4]` is the solution to a continuous
problem such as an ODE/SDE, or can be changed by a general `VariableRateJump`.
Thus, one must be careful to follow this rule when choosing rates.

In summary, if a particular jump process has a rate function that depends
explicitly or implicitly on a continuously changing quantity, you need to use a
[`VariableRateJump`](@ref).

## [Defining the Jumps Directly using a bounded `VariableRateJump`](@id VariableRateJumpWithBnds)

Assume that the infection rate is now decreasing over time. That is, when
individuals get infected, they immediately reach peak infectivity. The force of
infection then decreases exponentially to a basal level. In this case, we must
keep track of the time of infection events. Let the history ``H(t)`` contain the
timestamps of all ``I(t)`` active infections. The rate of infection is then

```math
\beta_1 S(t) I(t) + \alpha S(t) \sum_{t_i \in H(t)} \exp(-\gamma (t - t_i))
```

where ``\beta_1`` is the basal rate of infection, ``\alpha`` is the spike in the
rate of infection, and ``\gamma`` is the rate at which the spike decreases. Here,
we choose parameters such that infectivity rate due to a single infected
individual returns to the basal rate after spiking to ``\beta_1 + \alpha``. In
other words, we are modelling a situation in infected individuals gradually
become less infectious prior to recovering. Our parameters are then

```@example tut2
β1 = 0.001 / 1000.0
α = 0.1 / 1000.0
γ = 0.05
p1 = (β1, ν, α, γ)
```

We define a vector `H` to hold the timestamp of active infections. Then, we
define an infection reaction as a bounded `VariableRateJump`, requiring us to
again provide `rate` and `affect` functions, but also give functions that
calculate an upper-bound on the rate (`urate(u,p,t)`), an optional lower-bound
on the rate (`lrate(u,p,t)`), and a time window over which the bounds are valid
as long as any states these three rates depend on are unchanged
(`rateinterval(u,p,t)`). The lower- and upper-bounds of the rate should be valid
from the time they are computed `t` until `t + rateinterval(u, p, t)`:

```@example tut2
H = zeros(Float64, 10)
rate3(u, p, t) = p[1] * u[1] * u[2] + p[3] * u[1] * sum(exp(-p[4] * (t - _t)) for _t in H)
lrate = rate1              # β*S*I
urate = rate3
rateinterval(u, p, t) = 1 / (2 * urate(u, p, t))
function affect3!(integrator)
    integrator.u[1] -= 1     # S -> S - 1
    integrator.u[2] += 1     # I -> I + 1
    push!(H, integrator.t)
    nothing
end
jump3 = VariableRateJump(rate3, affect3!; lrate, urate, rateinterval)
```

Note that here we set the lower bound rate to be the normal SIR infection rate,
and set the upper bound rate equal to the new rate of infection (`rate3`). As
required for bounded `VariableRateJump`s, we have for any `s` in `[t,t + rateinterval(u,p,t)]` the bound `lrate(u,p,t) <= rate3(u,p,s) <= urate(u,p,t)`
will hold provided the dependent states `u[1]` and `u[2]` have not changed.

Next, we redefine the recovery jump's `affect!` such that a random infection is
removed from `H` for every recovery.

```@example tut2
rate4(u, p, t) = p[2] * u[2]         # ν*I
function affect4!(integrator)
    integrator.u[2] -= 1
    integrator.u[3] += 1
    length(H) > 0 && deleteat!(H, rand(1:length(H)))
    nothing
end
jump4 = ConstantRateJump(rate4, affect4!)
```

With the jumps defined, we can build a
[`DiscreteProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/discrete_types/).
Bounded `VariableRateJump`s over a `DiscreteProblem` can currently only be
simulated with `Coevolve`. The aggregator requires a dependency graph to
indicate when a given jump occurs and which other jumps in the system should
have their rate recalculated (i.e., their rate depends on states modified by
one occurrence of the first jump). This ensures that rates, rate bounds, and
rate intervals are recalculated when invalidated due to changes in `u`. For the
current example, both processes mutually affect each other, so we have

```@example tut2
dep_graph = [[1, 2], [1, 2]]
```

Here `dep_graph[2] = [1,2]` indicates that when the second jump occurs, both the
first and second jumps need to have their rates recalculated. We can then
construct our `JumpProblem` as before, specifying the `Coevolve` aggregator:

```@example tut2
prob = DiscreteProblem(u₀, tspan, p1)
jump_prob = JumpProblem(prob, Coevolve(), jump3, jump4; dep_graph)
```

We now have a problem that can be solved with `SSAStepper` to handle
time-stepping the `Coevolve` aggregator from jump to jump:

```@example tut2
sol = solve(jump_prob, SSAStepper())
plot(sol, label = ["S(t)" "I(t)" "R(t)"])
```

We see that the time-dependent infection rate leads to a lower peak of the
infection throughout the population.

Note that bounded `VariableRateJump`s over `DiscreteProblem`s can be quite
general. However, when handling rates that change according to an ODE/SDE
modified variable we will need a continuous integrator. One example is
discussed [below](@ref VariableRateJumpSect) in which we have a new reaction
added to the model with rate `p[2]*u[1]*u[4]` where `u[4]` is the solution of
an ODE. In such models, you will also need to be more careful in setting
rate bounds as they must be valid for the full coupled system's
dynamics.

## [Reducing Memory Use: Controlling Saving Behavior](@id save_positions_docs)

Note that jumps act via DifferentialEquations.jl's [callback
interface](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/),
which defaults to saving at each event. This is required in order to accurately
resolve every discontinuity exactly (and this is what allows for perfectly
vertical lines in plots!). However, in many cases when using jump problems, you
may wish to decrease the saving pressure given by large numbers of jumps. To do
this, you set the `save_positions` keyword argument to `JumpProblem`. Just like
for other
[callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/),
this is a tuple `(bool1, bool2)` which sets whether to save before or after a
jump. If we do not want to save at every jump, we would thus pass:

```@example tut2
prob = DiscreteProblem(u₀, tspan, p)
jump_prob = JumpProblem(prob, Direct(), jump, jump2; save_positions = (false, false))
```

Now the saving controls associated with the integrator should be specified, see the
main [SciML
Docs](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
for saving options. For example, we can use `saveat = 10.0` to save at an evenly
spaced grid:

```@example tut2
sol = solve(jump_prob, SSAStepper(); saveat = 10.0)

# we plot each solution component separately since
# the graph should no longer be a step function
plot(sol.t, sol[1, :]; marker = :o, label = "S(t)", xlabel = "t")
plot!(sol.t, sol[2, :]; marker = :x, label = "I(t)", xlabel = "t")
plot!(sol.t, sol[3, :]; marker = :d, label = "R(t)", xlabel = "t")
```

Notice that our plot (and solutions) are now defined at precisely the specified
time points. *It is important to note that interpolation of the solution object
will no longer be exact for a pure jump process, as the solution values at jump
times have not been stored. i.e for `t` a time we did not save at `sol(t)` will
no longer give the exact value of the solution at `t`.*

## Defining the Jumps Directly: Mixing `ConstantRateJump`/`VariableRateJump` and `MassActionJump`

Suppose we now want to add in to the original SIR model another jump that
cannot be represented as a mass action reaction. We can create a new
`ConstantRateJump` and simulate a hybrid system using both the `MassActionJump`
for the two original reactions, and the new `ConstantRateJump`. Let's suppose we
want to let susceptible people be born with the following jump rate:

```@example tut2
birth_rate(u, p, t) = 10.0 * u[1] / (200.0 + u[1]) + 10.0
function birth_affect!(integrator)
    integrator.u[1] += 1
    nothing
end
birth_jump = ConstantRateJump(birth_rate, birth_affect!)
```

We can then simulate the hybrid system as

```@example tut2
jump_prob = JumpProblem(prob, Direct(), mass_act_jump, birth_jump)
sol = solve(jump_prob, SSAStepper())
plot(sol; label = ["S(t)" "I(t)" "R(t)"])
```

Note that we can combine `MassActionJump`s, `ConstantRateJump`s and bounded
`VariableRateJump`s using the `Coevolve` aggregators.

## Adding Jumps to a Differential Equation

If we instead used some form of differential equation via an `ODEProblem`
instead of a `DiscreteProblem`, we can couple the jumps/reactions to the
differential equation. Let's define an ODE problem, where the continuous part
only acts on some new 4th component:

```@example tut2
using OrdinaryDiffEq
function f(du, u, p, t)
    du[4] = u[2] * u[3] / 1e5 - u[1] * u[4] / 1e5
    nothing
end
u₀ = [999.0, 10.0, 0.0, 100.0]
prob = ODEProblem(f, u₀, tspan, p)
```

Notice we gave the 4th component a starting value of 100.0, and used
floating-point numbers for the initial condition, since some solution components now
evolve continuously. The same steps as above will allow us to solve this hybrid
equation when using `ConstantRateJump`s, `MassActionJump`s, or
`VariableRateJump`s. For example, we can solve it using the `Tsit5()` method
via:

```@example tut2
jump_prob = JumpProblem(prob, Direct(), jump, jump2)
sol = solve(jump_prob, Tsit5())
plot(sol; label = ["S(t)" "I(t)" "R(t)" "u₄(t)"])
```

Note that in general, the ODE derivative `f(du, u, p, t)` could modify any
element in `du` which the jump rate functions depend on. In this section,
`f(du, u, p, t)` does not modify the jump rates so it is safe to couple them with any
type of jump and use any type of aggregator. However, the implementation does
not enforce this requirement, so one must be careful. Alternatively, when `f(du, u, p, t)` *does* modify variables that affect the jump rate, we have to
implement another strategy as described in the next [next Section](@ref
VariableRateJumpSect).

## [Adding a general VariableRateJump that Depends on a Continuously Evolving Variable](@id VariableRateJumpSect)

Now let's consider adding a reaction whose rate changes continuously with the
differential equation. To continue our example, let there be a new reaction
with rate depending on `u[4]` of the form ``u_4 \to u_4 + \textrm{I}``, with a
rate constant of `1e-2`:

```@example tut2
rate5(u, p, t) = 1e-2 * u[4]
function affect5!(integrator)
    integrator.u[2] += 1    # I -> I + 1
    nothing
end
jump5 = VariableRateJump(rate5, affect5!)
```

Notice, since `rate5` depends on a variable that evolves continuously, and hence
is not constant between jumps, *we must either use a general `VariableRateJump` without
upper/lower bounds or a bounded `VariableRateJump`*.

In the general case, solving the equation is exactly the same:

```@example tut2
u₀ = [999.0, 10.0, 0.0, 1.0]
prob = ODEProblem(f, u₀, tspan, p)
jump_prob = JumpProblem(prob, Direct(), jump, jump2, jump5)
sol = solve(jump_prob, Tsit5())
plot(sol; label = ["S(t)" "I(t)" "R(t)" "u₄(t)"])
```

*Note that general `VariableRateJump`s require using a continuous problem, like
an ODE/SDE/DDE/DAE problem, and using floating point initial conditions.*

Alternatively, the case of bounded `VariableRateJump` requires some maths.
First, we need to obtain the upper bounds of `rate5` at time `t` given `u`.
Note that `rate5` evolves according to `u[4]` which is a separable first order
differential equation of the form ``x' = b - a x`` with general solution:

```math
x(t) = - \frac{e^{-a t - c_1 a}}{a} + \frac{b}{a}
```

This is bounded by ``b / a`` which is too high for our purposes since it would
lead to a high rate of rejection during sampling. However, since the function
is increasing we can compute the upper bound given an interval ``\Delta t``
as following:

```math
\bar{x}(s) = x(t) \, e^{-a (t + \Delta t)} + \frac{b}{a} (1 - e^{- a (t + \Delta t)}) \text{ , } \forall s \in [t, t + \Delta t]
```

However, when ``a = 0`` the differential equation becomes ``x' = b`` whose solution is ``x(t) = b t``. In which case, we obtain a different upper bound given by:

```math
\bar{x}(s) = x(t) + b * (t + \Delta t) \text{ , } \forall s \in [t, t + \Delta t]
```

These expressions allow us to write the upper-bound and the rate interval in
Julia. In this example we use analytical boundaries for *illustration
purposes*. However, in some circumstances with complex model of many variables
it can be difficult to determine good a priori bounds on the ODE variables.
Moreover, numerical and analytical solutions are generally not guaranteed to
strictly satisfy the same bounds. In most cases the bounds should be close
enough. Thus, consider adding some slack on the bounds and approach complex
models with care.

```@example tut2
function urate2(u, p, t)
    if u[1] > 0
        1e-2 * max(u[4],
            (u[4] * exp(-1 * u[1] / 1e5) +
             (u[2] * u[3] / u[1]) * (1 - exp(-1 * u[1] / 1e5))))
    else
        1e-2 * (u[4] + 1 * u[2] * u[3] / 1e5)
    end
end
rateinterval2(u, p, t) = 1
```

We can then formulate the jump problem. The only aggregator that supports
bounded `VariableRateJump`s is `Coevolve`. We formulate and solve the
jump problem with this aggregator. `Coevolve` can be formulated as either
a discrete or continuous problem. In this case, we must formulate the problem
as continuous as it depends on a continuous variable.

```@example tut2
jump6 = VariableRateJump(rate5, affect5!; urate = urate2, rateinterval = rateinterval2)
dep_graph2 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
jump_prob = JumpProblem(prob, Coevolve(), jump, jump2, jump6; dep_graph = dep_graph2)
sol = solve(jump_prob, Tsit5())
plot(sol; label = ["S(t)" "I(t)" "R(t)" "u₄(t)"])
```

We obtain the same solution as with `Direct`, but `Coevolve` runs faster
because it doesn't need to compute the derivative of `rate5`. Each aggregator
faces a different trade-off, so the the choice of best aggregator will depend
on the problem at hand. `Coevolve` requires a good understanding of the
equations involved, passing a wrong boundary can result in silent bugs.

Lastly, we are not restricted to ODEs. For example, we can solve the same jump
problem except with multiplicative noise on `u[4]` by using an `SDEProblem`
instead:

```@example tut2
using StochasticDiffEq
function g(du, u, p, t)
    du[4] = 0.1u[4]
end
prob = SDEProblem(f, g, [999.0, 1.0, 0.0, 1.0], (0.0, 250.0), p)
jump_prob = JumpProblem(prob, Direct(), jump, jump2, jump5)
sol = solve(jump_prob, SRIW1())
plot(sol; label = ["S(t)" "I(t)" "R(t)" "u₄(t)"])
```

For more details about general `VariableRateJump`s see [Defining a Variable Rate
Jump](@ref).

## RegularJumps and τ-Leaping

The previous parts described how to use `ConstantRateJump`s, `MassActionJump`s,
and `VariableRateJump`s. However, in many cases one does not require the
exactness of stepping to every jump time. Instead, regular jumping (i.e.,
τ-leaping) allows pooling jumps together, and performing larger updates in a
statistically-correct but more efficient manner. The trade-off is the
introduction of a time-discretization error due to the time-stepping, but one
that is controlled and convergent as the time-step is reduced to zero.

Let's see how to define the SIR model in terms of a `RegularJump`. We need two
functions, `rate` and `change!`. `rate` is a vector equation which computes the
rates of each jump process

```@example tut2
function rate(out, u, p, t)
    out[1] = p[1] * u[1] * u[2]   # β * S * I
    out[2] = p[2] * u[2]          # ν * I
    nothing
end
```

We then define a function that given a vector storing the number of times each
jump occurs during a time-step, `counts`, calculates the change in the state,
`du`. For the SIR example, we do this by multiplying `counts` by a matrix that
encodes the change in the state due to one occurrence of each reaction (i.e., the
net stoichiometry matrix). Below `c[i,j]` gives the change in `u[i]` due to the
`j`th jump:

```@example tut2
c = zeros(3, 2)
# S + I --> I
c[1, 1] = -1    # S -> S - 1
c[2, 1] = 1     # I -> I + 1

# I --> R
c[2, 2] = -1    # I -> I - 1
c[3, 2] = 1     # R -> R + 1

function change(du, u, p, t, counts, mark)
    mul!(du, c, counts)
    nothing
end
```

We are now ready to create our `RegularJump`, passing in the rate function,
change function, and the number of jumps being encoded

```@example tut2
rj = RegularJump(rate, change, 2)
```

From there we build a `JumpProblem`

```@example tut2
u₀ = [1000.0, 50.0, 0.0]
prob = DiscreteProblem(u₀, tspan, p)
jump_prob = JumpProblem(prob, Direct(), rj)
```

Note that when a `JumpProblem` has a `RegularJump`, τ-leaping algorithms are
required for simulating it. This is detailed on the [jump solvers page](@ref
jump_solve). One such algorithm is `TauLeaping` from StochasticDiffEq.jl, which
we use as follows:

```@example tut2
sol = solve(jump_prob, TauLeaping(); dt = 0.001)
plot(sol; label = ["S(t)" "I(t)" "R(t)"])
```
