using JumpProcesses, DiffEqBase, OrdinaryDiffEq, Statistics
using Test
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> u
affect! = function (integrator)
    integrator.u += 1
end
jump = ConstantRateJump(rate, affect!)

rate = (u, p, t) -> 0.5u
affect! = function (integrator)
    integrator.u -= 1
end
jump2 = ConstantRateJump(rate, affect!)

prob = DiscreteProblem(1.0, (0.0, 3.0))
jump_prob = JumpProblem(prob, Direct(), jump)

sol = solve(jump_prob, FunctionMap(); rng)

# using Plots; plot(sol)

prob = DiscreteProblem(10.0, (0.0, 3.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2)

sol = solve(jump_prob, FunctionMap(); rng)

# plot(sol)

nums = Int[]
@time for i in 1:10000
    local jump_prob = JumpProblem(prob, Direct(), jump, jump2)
    local sol = solve(jump_prob, FunctionMap(); rng)
    push!(nums, sol.u[end])
end

@test mean(nums) - 45 < 1

prob = DiscreteProblem(1.0, (0.0, 3.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2)

sol = solve(jump_prob, FunctionMap(); rng)

nums = Int[]
@time for i in 1:10000
    local jump_prob = JumpProblem(prob, Direct(), jump, jump2)
    local sol = solve(jump_prob, FunctionMap(); rng)
    push!(nums, sol.u[2])
end

@test sum(nums .== 0) / 10000 - 0.33 < 0.02
