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
jump_prob = JumpProblem(prob,jump)

sol = solve(jump_prob,Discrete(apply_map=false))

using Plots; plot(sol,plotdensity=1000)

prob = DiscreteProblem(10.0,(0.0,3.0))
jump_prob = JumpProblem(prob,jump,jump2)

sol = solve(jump_prob,Discrete(apply_map=false))

plot(sol,plotdensity=1000)




nums = Int[]
@time for i in 1:100000
  sol = solve(jump_prob,Discrete(apply_map=false))
  push!(nums,sol[end])
end
mean(nums)


js = JumpSet(jump,jump2)

constant_jumps = js.constant_jumps

DiscreteCallback(CompoundConstantRateJump(0.0,1.0,3.0,constant_jumps))


rates = ((c.rate for c in constant_jumps)...)
affects! = ((c.affect! for c in constant_jumps)...)
cur_rates = Vector{Float64}(length(rates))
sum_rate,next_jump = JumpDiffEq.time_to_next_jump(0.0,1.0,rates,cur_rates)
