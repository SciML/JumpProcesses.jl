using JumpProcesses, DiffEqBase, SciMLBase
using Test
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> u[1]
affect! = function (integrator)
    return integrator.u[1] += 1
end
jump = ConstantRateJump(rate, affect!)

rate = (u, p, t) -> 0.5u[1]
affect! = function (integrator)
    return integrator.u[1] -= 1
end
jump2 = ConstantRateJump(rate, affect!)

prob = DiscreteProblem([10.0], (0.0, 3.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)

integrator = init(jump_prob, SSAStepper())
step!(integrator)
integrator.u[1]

# test different saving behaviors

sol = solve(jump_prob, SSAStepper())
@test SciMLBase.successful_retcode(sol)
@test sol.t[begin] == 0.0
@test sol.t[end] == 3.0

sol = solve(jump_prob, SSAStepper(), save_end = false)
@test sol.t[begin] == 0.0
@test sol.t[end] < 3.0

sol = solve(jump_prob, SSAStepper(), save_start = false)
@test sol.t[begin] > 0.0
@test sol.t[end] == 3.0

jump_prob = JumpProblem(
    prob, Direct(), jump, jump2, save_positions = (false, false);
    rng = rng
)
sol = solve(jump_prob, SSAStepper(), save_start = false, save_end = false)
@test isempty(sol.t) && isempty(sol.u)

sol = solve(jump_prob, SSAStepper(), saveat = 0.0:0.1:2.9)
@test sol.t == collect(0.0:0.1:3.0)
