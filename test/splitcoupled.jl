using DiffEqJump, DiffEqBase, OrdinaryDiffEq
using Base.Test


rate = (t,u) -> 100.*u[1]
affect! = function (integrator)
  integrator.u[1] += 1
end
jump1 = ConstantRateJump(rate,affect!)

prob = DiscreteProblem([10],(0.0,50.0))
prob_control = DiscreteProblem([10],(0.0,50.0))

jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump1)

coupling_map = [(1, 1)]
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)

@time sol =  solve(coupled_prob,Discrete(apply_map=false))
@time solve(jump_prob,Discrete(apply_map=false))
@test [s[1]-s[2] for s in sol.u] == zeros(length(sol.t)) # coupling two copies of the same process should give zero
