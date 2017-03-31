using DiffEqJump, DiffEqBase, OrdinaryDiffEq
using Base.Test

rate = (t,u) -> (0.1/1000.0)*u[1]*u[2]
affect! = function (integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump = ConstantRateJump(rate,affect!;save_positions=(false,true))

rate = (t,u) -> 0.01u[2]
affect! = function (integrator)
  integrator.u[2] -= 1
  integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate,affect!;save_positions=(false,true))


prob = DiscreteProblem([999.0,1.0,0.0],(0.0,250.0))
jump_prob = JumpProblem(prob,Direct(),jump,jump2)
sol = solve(jump_prob,Discrete(apply_map=false))

using Plots; plot(sol)

nums = Int[]
@time for i in 1:1000
  sol = solve(jump_prob,Discrete(apply_map=false))
  push!(nums,sol[end][1])
end
mean(nums)
