using JumpDiffEq, DiffEqBase, OrdinaryDiffEq
using Base.Test

rate = (t,u) -> u
affect! = function (integrator)
  integrator.u += 1
end
jump = ConstantRateJump(rate,affect!;save_positions=(false,true))

rate = (t,u) -> 0.5u
affect! = function (integrator)
  integrator.u -= 1
end
jump2 = ConstantRateJump(rate,affect!;save_positions=(false,true))


prob = DiscreteProblem(1.0,(0.0,3.0))
jump_prob = JumpProblem(prob,Direct(),jump)

sol = solve(jump_prob,Discrete(apply_map=false))

# using Plots; plot(sol)

prob = DiscreteProblem(10.0,(0.0,3.0))
jump_prob = JumpProblem(prob,Direct(),jump,jump2)

sol = solve(jump_prob,Discrete(apply_map=false))

# plot(sol)

nums = Int[]
@time for i in 1:100000
  jump_prob = JumpProblem(prob,Direct(),jump,jump2)
  sol = solve(jump_prob,Discrete(apply_map=false))
  push!(nums,sol[end])
end
mean(nums)
