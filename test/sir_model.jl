using DiffEqJump, DiffEqBase, OrdinaryDiffEq
using Test
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> (0.1 / 1000.0) * u[1] * u[2]
affect! = function (integrator)
    integrator.u[1] -= 1
    integrator.u[2] += 1
end
jump = ConstantRateJump(rate, affect!)

rate = (u, p, t) -> 0.01u[2]
affect! = function (integrator)
    integrator.u[2] -= 1
    integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate, affect!)

prob = DiscreteProblem([999.0, 1.0, 0.0], (0.0, 250.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)
integrator = init(jump_prob, FunctionMap())

condition(u, t, integrator) = t == 100
function purge_affect!(integrator)
    integrator.u[2] ÷= 10
    reset_aggregated_jumps!(integrator)
end
cb = DiscreteCallback(condition, purge_affect!, save_positions = (false, false))
sol = solve(jump_prob, FunctionMap(), callback = cb, tstops = [100])
sol = solve(jump_prob, SSAStepper(), callback = cb, tstops = [100])
