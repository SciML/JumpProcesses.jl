using DiffEqJump, DiffEqBase
using Test

rate = (u, p, t) -> u[1]
affect! = function (integrator)
    integrator.u[1] -= 1
    integrator.u[2] += 1
end
jump = ConstantRateJump(rate, affect!)

prob = DiscreteProblem([0.0, 0.0], (0.0, 10.0))
jump_prob = JumpProblem(prob, Direct(), jump)

sol = solve(jump_prob, SSAStepper())

@test sol.t == [0.0, 10.0]
@test sol.u == [[0.0, 0.0], [0.0, 0.0]]

condition(u,t,integrator) = t == 5
function fuel_affect!(integrator)
  integrator.u[1] += 100
  reset_aggregated_jumps!(integrator)
end
cb = DiscreteCallback(condition, fuel_affect!, save_positions=(false, true))

sol = solve(jump_prob, SSAStepper(), callback=cb, tstops=[5])

@test sol.t[1:2] == [0.0, 5.0] # no jumps between t=0 and t=5
@test sol(5 + 1e-10) == [100, 0] # state just after fueling before any decays can happen
