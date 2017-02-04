using JumpDiffEq, DiffEqBase, OrdinaryDiffEq
using Base.Test

rate = (t,u) -> u
affect! = function (integrator)
  integrator.u += 1
end
jump = ConstantRateJump(rate,affect!;save_positions=(false,true))

prob = DiscreteProblem(1.0,(0.0,3.0))
jump_prob = JumpProblem(prob,jump)

sol = solve(jump_prob,Discrete(apply_map=false))

using Plots; plot(sol,plotdensity=1000)
