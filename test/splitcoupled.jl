using DiffEqJump, DiffEqBase, OrdinaryDiffEq, StochasticDiffEq
using Base.Test


rate = (t,u) -> 1.*u[1]
affect! = function (integrator)
  integrator.u[1] = 1.
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


rate = (t,u) -> 1.
affect! = function (integrator)
  integrator.u[1] = 1.
end
jump1 = ConstantRateJump(rate,affect!)
rate = (t,u) -> 2.
jump2 = ConstantRateJump(rate,affect!)

f = function (t,u,du)
  du[1] = u[1]
end
g = function (t,u,du)
  du[1] = 0.1
end

# Jump ODE to jump ODE
prob = ODEProblem(f,[1.],(0.0,1.0))
prob_control = ODEProblem(f,[1.],(0.0,1.0))
jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump2)
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)
sol =  solve(coupled_prob,Tsit5())
@test mean([abs(s[1]-s[2]) for s in sol.u])<=5.


# Jump SDE to Jump SDE
prob = SDEProblem(f,g,[1.],(0.0,1.0))
prob_control = SDEProblem(f,g,[1.],(0.0,1.0))
jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump1)
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)
sol =  solve(coupled_prob,SRIW1())
@test mean([abs(s[1]-s[2]) for s in sol.u])<=5.


# Jump SDE to Jump ODE
prob = ODEProblem(f,[1.],(0.0,1.0))
prob_control = SDEProblem(f,g,[1.],(0.0,1.0))
jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump1)
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)
sol =  solve(coupled_prob,SRIW1())
@test mean([abs(s[1]-s[2]) for s in sol.u])<=5.

# Jump SDE to Discrete
rate = (t,u) -> 1.
affect! = function (integrator)
  integrator.u[1] += 1.
end
prob = DiscreteProblem([1.],(0.0,1.0))
prob_control = SDEProblem(f,g,[1.],(0.0,1.0))
jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump1)
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)
sol =  solve(coupled_prob,SRIW1())
