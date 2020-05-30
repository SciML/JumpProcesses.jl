using DiffEqJump, DiffEqBase, OrdinaryDiffEq, StochasticDiffEq, Statistics
using Test


rate = (u,p,t) -> 1.0*u[1]
affect! = function (integrator)
  integrator.u[1] = 1.0
end
jump1 = ConstantRateJump(rate,affect!)

prob = DiscreteProblem([10],(0.0,50.0))
prob_control = DiscreteProblem([10],(0.0,50.0))

jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump1)

coupling_map = [(1, 1)]
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)

@time sol =  solve(coupled_prob,FunctionMap())
@time solve(jump_prob,FunctionMap())
@test [s[1]-s[2] for s in sol.u] == zeros(length(sol.t)) # coupling two copies of the same process should give zero


rate = (u,p,t) -> 1.0
affect! = function (integrator)
  integrator.u[1] = 1.0
end
jump1 = ConstantRateJump(rate,affect!)
rate = (u,p,t) -> 2.0
jump2 = ConstantRateJump(rate,affect!)

f = function (du,u,p,t)
  du[1] = u[1]
end
g = function (du,u,p,t)
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
sol =  solve(coupled_prob,SRIW1(),seed=UInt64(1))
@test mean([abs(s[1]-s[2]) for s in sol.u])<=5.

# Jump SDE to Jump ODE
prob = ODEProblem(f,[1.],(0.0,1.0))
prob_control = SDEProblem(f,g,[1.],(0.0,1.0))
jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump1)
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)
sol =  solve(coupled_prob,SRIW1(),seed=UInt64(1))
@test mean([abs(s[1]-s[2]) for s in sol.u])<=5.

# Jump SDE to Discrete
rate = (u,p,t) -> 1.
affect! = function (integrator)
  integrator.u[1] += 1.
end
prob = DiscreteProblem([1.],(0.0,1.0))
prob_control = SDEProblem(f,g,[1.],(0.0,1.0))
jump_prob = JumpProblem(prob,Direct(),jump1)
jump_prob_control = JumpProblem(prob_control,Direct(),jump1)
coupled_prob = SplitCoupledJumpProblem(jump_prob,jump_prob_control,Direct(),coupling_map)
sol =  solve(coupled_prob,SRIW1(),seed=UInt64(1))


# test mass action jumps coupled to ODE
# 0 -> A (stochasic) and A -> 0 (ODE)
rate        = [100.]
react_stoch = [Vector{Pair{Int,Int}}()]
net_stoch   = [[1 => 1]]
majumps     = MassActionJump(rate, react_stoch, net_stoch)
f = function (du,u,p,t)
  du[1] = -1.0*u[1]
end
odeprob     = ODEProblem(f,[10.0],(0.0,10.0))
jump_prob   = JumpProblem(odeprob, Direct(), majumps, save_positions=(false,false))
Nsims = 8000
Amean = 0.
for i in 1:Nsims
  global Amean
  sol    = solve(jump_prob,Tsit5(),saveat=10.)
  Amean += sol[1,end]
end
Amean /= Nsims
actmean = 100. + (10.0-100.0)*exp(-1.0*10.0)
#println(abs(Amean-actmean)/actmean)
@test abs(actmean - Amean) < .02 * actmean
