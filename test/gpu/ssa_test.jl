using JumpProcesses, DiffEqBase, SciMLBase, Plots, CUDA
using Test, LinearAlgebra
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> u[1]
affect! = function (integrator)
    integrator.u[1] += 1
end
jump = ConstantRateJump(rate, affect!)

rate = (u, p, t) -> 0.5u[1]
affect! = function (integrator)
    integrator.u[1] -= 1
end
jump2 = ConstantRateJump(rate, affect!)

prob = DiscreteProblem([10.0], (0.0, 3.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)

integrator = init(jump_prob, SSAStepper())
step!(integrator)
integrator.u[1]

# test different saving behaviors

sol = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleGPUKernel(), 
            trajectories=100, saveat=1.0)
plot(sol)
