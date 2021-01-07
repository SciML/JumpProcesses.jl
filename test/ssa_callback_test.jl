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

# test that callback initializer/finalizer is called and add_tstop! works as expected
function fuel_init!(cb,u,t,integrator)
  for tstop in 1:9
    add_tstop!(integrator, tstop)
  end
end
finalizer_called = 0
fuel_finalize(cb, u, t, integrator) = global finalizer_called += 1

cb2 = DiscreteCallback(condition, fuel_affect!, initialize=fuel_init!, finalize=fuel_finalize)
sol = solve(jump_prob, SSAStepper(), callback=cb2, tstops=[])
for tstop in 1:9
  @test tstop ∈ sol.t
end
@test finalizer_called == 1
