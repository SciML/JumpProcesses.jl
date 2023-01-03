# FAQ

## My simulation is really slow and/or using a lot of memory, what can I do?
Exact methods simulate every jump, and by default save the state before and
after each jump. To reduce memory use, use `save_positions = (false, false)` in
the `JumpProblem` constructor as described [earlier](@ref save_positions_docs)
to turn off saving the system state before and after every jump. Combined with
use of `saveat` in the call to `solve`, to specify the specific times at which
to save the state, this can dramatically reduce memory usage.

While `Direct` is often fastest for systems with 10 or less `ConstantRateJump`s
and/or `MassActionJump`s, if your system has many jumps or one jump occurs most
frequently, other stochastic simulation algorithms may be faster. See [Jump
Aggregators for Exact Simulation](@ref) and the subsequent sections there for
guidance on choosing different SSAs (called aggregators in JumpProcesses). For
systems with bounded `VariableRateJump`s using `Coevolve` with `SSAStepper`
instead of an ODE/SDE time stepper can give a significant performance boost.

## When running many consecutive simulations, for example within an `EnsembleProblem` or loop, how can I update `JumpProblem`s?

In [Remaking `JumpProblem`s](@ref) we show how to modify parameters, the initial
condition, and other components of a generated `JumpProblem`. This can be useful
when trying to call `solve` many times while avoiding reallocations of the
internal aggregators for each new parameter value or initial condition.

## How can I define collections of many different jumps and pass them to `JumpProblem`?

We can use `JumpSet`s to collect jumps together, and then pass them into
`JumpProblem`s directly. For example, using a `MassActionJump` and
`ConstantRateJump` defined in the [second tutorial](@ref ssa_tutorial), we can
write

```julia
jset = JumpSet(mass_act_jump, birth_jump)
jump_prob = JumpProblem(prob, Direct(), jset)
sol = solve(jump_prob, SSAStepper())
```

If you have many jumps in tuples or vectors it is easiest to use the keyword
argument-based constructor:
```julia
cj1 = ConstantRateJump(rate1, affect1!)
cj2 = ConstantRateJump(rate2, affect2!)
cjvec = [cj1, cj2]

vj1 = VariableRateJump(rate3, affect3!)
vj2 = VariableRateJump(rate4, affect4!)
vjtuple = (vj1, vj2)

jset = JumpSet(; constant_jumps = cjvec, variable_jumps = vjtuple,
                 massaction_jumps = mass_act_jump)
```

## How can I set the random number generator used in the jump process sampling algorithms (SSAs)?

Random number generators can be passed to `JumpProblem` via the `rng` keyword
argument. Continuing the previous example:

```julia
#] add RandomNumbers
using RandomNumbers
jprob = JumpProblem(dprob, Direct(), maj,
                    rng = Xorshifts.Xoroshiro128Star(rand(UInt64)))
```
uses the `Xoroshiro128Star` generator from
[RandomNumbers.jl](https://github.com/JuliaRandom/RandomNumbers.jl).

On version 1.7 and up JumpProcesses uses Julia's builtin random number generator by
default. On versions below 1.7 it uses `Xoroshiro128Star`.

## What are these aggregators and aggregations in JumpProcesses?

JumpProcesses provides a variety of methods for sampling the time the next
`ConstantRateJump`, `MassActionJump`, or `VariableRateJump` occurs, and which
jump type happens at that time. These methods are examples of stochastic
simulation algorithms (SSAs), also known as Gillespie methods, Doob's method, or
Kinetic Monte Carlo methods. These are all names for jump (or point) processes
simulation methods used across the biology, chemistry, engineering, mathematics,
and physics literature. In the JumpProcesses terminology we call such methods
"aggregators", and the cache structures that hold their basic data
"aggregations". See [Jump Aggregators for Exact Simulation](@ref) for a list of
the available SSA aggregators.

## How should jumps be ordered in dependency graphs?
Internally, JumpProcesses SSAs (aggregators) order all `MassActionJump`s first,
then all `ConstantRateJumps` and/or `VariableRateJumps`. i.e. in the example

```julia
using JumpProcesses
rs = [[1 => 1], [2 => 1]]
ns = [[1 => -1, 2 => 1], [1 => 1, 2 => -1]]
p = [1.0, 0.0]
maj = MassActionJump(rs, ns; param_idxs=[1, 2])
rate1(u, p, t) = u[1]
function affect1!(integrator)
    u[1] -= 1
end
cj1 = ConstantRateJump(rate1, affect1)
rate2(u, p, t) = u[2]
function affect2!(integrator)
    u[2] -= 1
end
cj2 = ConstantRateJump(rate2, affect2)
jset = JumpSet(; constant_jumps=[cj1, cj2], massaction_jump=maj)
```
The four jumps would be ordered by the first jump in `maj`, the second jump in
`maj`, `cj1`, and finally `cj2`. Any user-generated dependency graphs should
then follow this ordering when assigning an integer id to each jump.

See also [Jump Aggregators Requiring Dependency Graphs](@ref) for
more on dependency graphs needed for the various SSAs.

## How do I use callbacks with jump simulations?

Callbacks can be used with `ConstantRateJump`s, `MassActionJump`s, and
`VariableRateJump`s. When solving a pure jump system with `SSAStepper`, only
discrete callbacks can be used (otherwise a different time stepper is needed).
When using an ODE or SDE time stepper any callback should work.

*Note, when modifying `u` or `p` within a callback, you must call
[`reset_aggregated_jumps!`](@ref) after making updates.* This ensures that the
underlying jump simulation algorithms know to reinitialize their internal data
structures. Leaving out this call will lead to incorrect behavior!

A simple example that uses a `MassActionJump` and changes the parameters at a
specified time in the simulation using a `DiscreteCallback` is
```julia
using JumpProcesses
rs = [[1 => 1], [2 => 1]]
ns = [[1 => -1, 2 => 1], [1 => 1, 2 => -1]]
p = [1.0, 0.0]
maj = MassActionJump(rs, ns; param_idxs=[1, 2])
u₀ = [100, 0]
tspan = (0.0, 40.0)
dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), maj)
pcondit(u, t, integrator) = t == 20.0
function paffect!(integrator)
    integrator.p[1] = 0.0
    integrator.p[2] = 1.0
    reset_aggregated_jumps!(integrator)
    nothing
end
sol = solve(jprob, SSAStepper(), tstops=[20.0], callback=DiscreteCallback(pcondit, paffect!))
```
Here at time `20.0` we turn off production of `u[2]` while activating production
of `u[1]`, giving

![callback_gillespie](assets/callback_gillespie.png)


## How can I access earlier solution values in callbacks?
When using an ODE or SDE time-stepper that conforms to the [integrator
interface](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/) one
can simply use `integrator.uprev`. For efficiency reasons, the pure jump
[`SSAStepper`](@ref) integrator does not have such a field. If one needs
solution components at earlier times one can save them within the callback
condition by making a functor:
```julia
# stores the previous value of u[2] and represents the callback functions
mutable struct UprevCondition{T}
     u2::T
end

# condition
function (upc::UprevCondition)(u, t, integrator)
    # condition for the callback is that the new value of u[2]
    # is smaller than the previous value
    condit = u[2] - upc.u2 < 0

    # save the new value as the previous value
    upc.u2 = u[2]

    condit
end

# affect!
function (upc::UprevCondition)(integrator)
    integrator.u[4] -= 1
    nothing
end

upc = UprevCondition(u0[2])
cb = DiscreteCallback(upc, upc)
```
