## Coupling Jump Problems

THIS NEEDS UPDATING

In many applications one is interested in coupling two stochastic processes.
This has applications in Monte Carlo simulations and sensitivity analysis, for
example. Currently, the coupling that is implemented for jump processes is known
as the split coupling. The split coupling couples two jump processes by coupling
the underlying Poisson processes driving the jump components.

Suppose `prob` and `prob_control` are two problems we wish to couple. Then the
coupled problem is obtained by

```@example tut3
prob_coupled = SplitCoupledJumpProblem(jump_prob, jump_prob_control, Direct(), coupling_map)
```

Here, `coupling_map` specifies which jumps to couple. If `(j,i)` is in
`coupling_map`, then the `i`th jump in `prob` will be coupled to the `j`th jump
in `prob_control`. Note that currently `SplitCoupledJumpProblem` is only
implemented for `ConstantRateJump` problems.

As an example, consider a doubly stochastic Poisson process, that is, a Poisson
process whose rate is itself a stochastic process. In particular, we will take
the rate to randomly switch between zero and `10` at unit rates:

```@example tut3
rate(u, p, t) = 10 * u[2]
affect!(integrator) = integrator.u[1] += 1
jump1 = ConstantRateJump(rate, affect!)

rate(u, p, t) = u[2]
affect!(integrator) = (integrator.u[2] -= 1; integrator.u[3] += 1)
jump2 = ConstantRateJump(rate, affect!)

rate(u, p, t) = u[3]
affect!(integrator) = (integrator.u[2] += 1; integrator.u[3] -= 1)
jump3 = ConstantRateJump(rate, affect!)

prob = DiscreteProblem(u0, tspan)
jump_prob = JumpProblem(prob, Direct(), jump1, jump2, jump3)
```

The doubly stochastic Poisson process has two sources of randomness: one due to
the Poisson process, and another due to random evolution of the rate. This is
typical of many multiscale stochastic processes appearing in applications, and
it is often useful to compare such a process to one obtained by removing one
source of randomness. In present context, this means looking at an ODE with
constant jump rates, where the deterministic evolution between jumps is given by
the expected value of the Poisson process:

```@example tut3
function f(du, u, p, t)
    du[1] = 10 * u[2]
    du[2] = 0
    du[3] = 0
end
prob_control = ODEProblem(f, [0.0, 0.0, 1.0], tspan)
jump_prob_control = JumpProblem(prob_control, Direct(), jump2, jump3)
```

Let's couple the two problems by coupling the jumps corresponding to the
switching of the rate:

```@example tut3
coupling_map = [(2, 1), (3, 2)]
prob_coupled = SplitCoupledJumpProblem(jump_prob, jump_prob_control, Direct(), coupling_map)
```

Now `prob_coupled` can be solved like any other `JumpProblem`:

```@example tut3
sol = solve(prob_coupled, Tsit5())
```
