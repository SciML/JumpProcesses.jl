using JumpProcesses, DiffEqBase
using Random
using Test

rate = (u, p, t) -> u[1]
affect! = integrator -> (integrator.u[1] += 1)
jump = ConstantRateJump(rate, affect!)
prob = DiscreteProblem([10.0], (0.0, 1.0))
jump_prob = JumpProblem(prob, Direct(), jump;
    rng = MersenneTwister(12345), save_positions = (false, false))

integrator = init(jump_prob, SSAStepper())
step!(integrator)
@test integrator.u[1] == 11.0

sol = solve(jump_prob, SSAStepper())
@test sol.t[end] == 1.0
@test sol.u[end][1] == 21.0
