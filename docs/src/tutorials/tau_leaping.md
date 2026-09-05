# [Tau-leaping with mass-action and regular jumps](@id tau_leaping_tutorial)

Tau-leaping approximates many individual reaction events by a single update over
an interval. It introduces discretization error in addition to Monte Carlo
sampling error. Compare ensembles and repeat with smaller steps or tighter
step-selection parameters when assessing accuracy.

## Adaptive mass-action methods

JumpProcesses' adaptive leaping solvers accept a `MassActionJump` directly. Here a
reversible conversion is coupled to a slower irreversible reaction:

```@example tau
using JumpProcesses

maj = MassActionJump(
    [100.0, 100.0, 0.1],
    [[1 => 1], [2 => 1], [1 => 1]],
    [[1 => -1, 2 => 1], [1 => 1, 2 => -1], [1 => -1, 3 => 1]]
)
prob = DiscreteProblem([100.0, 100.0, 0.0], (0.0, 1.0))
jprob = JumpProblem(prob, PureLeaping(), maj)

explicit_sol = solve(jprob, SimpleExplicitTauLeaping(; epsilon = 0.01); saveat = 0.1)
implicit_sol = solve(jprob, SimpleImplicitTauLeaping(); saveat = 0.1)
trapezoidal_sol = solve(jprob, SimpleTrapezoidalLeaping(); saveat = 0.1)
switching_sol = solve(jprob, SimpleAdaptiveTauLeaping(); saveat = 0.1)
nothing # hide
```

The explicit method adapts its leap size. The implicit and trapezoidal methods
also adapt their leap sizes, and perform a nonlinear solve at each step. The
switching method chooses an explicit or implicit step from its stiffness test;
"adaptive" in its name refers to this switching, not just step-size adaptation.
An implicit step costs more, so it is useful when stiffness restricts explicit
steps severely.

Select the alternative implicit formulation and stiffness test as follows:

```@example tau
alg = SimpleAdaptiveTauLeaping(;
    implicit_alg = SimpleTrapezoidalLeaping(), eigenvalue_check = true
)
sol = solve(jprob, alg; saveat = 0.1)

using Test
for solution in (explicit_sol, implicit_sol, trapezoidal_sol, switching_sol, sol)
    @test solution.t == collect(0.0:0.1:1.0)
    @test all(u -> all(>=(0), u), solution.u)
    @test all(u -> sum(u) ≈ 200.0, solution.u)
end
nothing # hide
```

These solvers require a pure mass-action model over a `DiscreteProblem`; do not
add a `RegularJump` or separately aggregated jumps to this problem.

## Reusing the mass-action problem with StochasticDiffEq

The same `jprob` also works with `SimpleTauLeaping` and StochasticDiffEq's
leaping solvers. No conversion to a user-written `RegularJump` is needed:

```@example tau
using StochasticDiffEq

fixed_sol = solve(jprob, SimpleTauLeaping(); dt = 0.001)
tau_sol = solve(jprob, TauLeaping(); dt = 0.001)
cao_sol = solve(jprob, CaoTauLeaping(); dt = 0.001)
sde_implicit_sol = solve(jprob, ImplicitTauLeaping(); dt = 0.001, adaptive = false)
theta_sol = solve(
    jprob, ThetaTrapezoidalTauLeaping(); dt = 0.001, adaptive = false
)

for solution in (fixed_sol, tau_sol, cao_sol, sde_implicit_sol, theta_sol)
    @test successful_retcode(solution)
    @test solution.t[end] ≈ 1.0
    @test all(u -> sum(u) ≈ 200.0, solution.u)
end
nothing # hide
```

## More general rates with RegularJump

`SimpleTauLeaping` and StochasticDiffEq's tau-leaping solvers accept
`RegularJump(rate!, change!, numjumps)`. `rate!` writes every propensity into its
output, and `change!` writes the **increment**, not the new state, associated with
the reaction counts. For example, a conversion with a saturating rate from the
first species to the second is:

```@example regular_tau
using JumpProcesses, StochasticDiffEq

function rate!(out, u, p, t)
    out[1] = p * u[1] / (1 + u[1])
    return nothing
end

function change!(du, u, p, t, counts, mark)
    du[1] = -counts[1]
    du[2] = counts[1]
    return nothing
end

rj = RegularJump(rate!, change!, 1)
prob = DiscreteProblem([1000.0, 0.0], (0.0, 1.0), 100.0)
jprob = JumpProblem(prob, PureLeaping(), rj)

fixed_sol = solve(jprob, SimpleTauLeaping(); dt = 0.01)
adaptive_sol = solve(jprob, TauLeaping(); dt = 0.01)
cao_sol = solve(jprob, CaoTauLeaping(); dt = 0.01)
implicit_sol = solve(jprob, ImplicitTauLeaping(); dt = 0.01, adaptive = false)
theta_sol = solve(
    jprob, ThetaTrapezoidalTauLeaping(; theta = 0.5); dt = 0.01, adaptive = false
)

using Test
for solution in (fixed_sol, adaptive_sol, cao_sol, implicit_sol, theta_sol)
    @test solution.t[end] ≈ 1.0
    @test all(u -> sum(u) ≈ 1000.0, solution.u)
end
nothing # hide
```

The same `RegularJump` and `JumpProblem` are reused here. The implicit examples
explicitly select fixed stepping; supplying `dt` alone does not disable adaptive
stepping in StochasticDiffEq. The `change!` function must also make sense for the
non-integer rate/drift inputs used by implicit solvers; a linear stoichiometric
update as above satisfies this requirement.

`MassActionJump` is the common input for all the leaping methods above.
`RegularJump` is an additional option for custom kinetics in `SimpleTauLeaping`
and StochasticDiffEq. JumpProcesses' adaptive mass-action solvers need the
reaction-order and stoichiometric information provided by `MassActionJump`. See
[the solver table](@ref jump_solve) and [GPU support](@ref gpu_ensembles) before
switching algorithms or ensemble backends.
