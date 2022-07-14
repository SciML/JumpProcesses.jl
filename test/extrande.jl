using DiffEqBase, JumpProcesses, OrdinaryDiffEq, Test
using StableRNGs
rng = StableRNG(12345)

rate = (u,p,t) -> t
affect! = (integrator) -> (integrator.u[1] = integrator.u[1]+1)
rbound = (u,p,t) -> (t + 0.1)
rwindow = (u,p,t) -> 0.1
jump = VariableRateJump(rate,affect!,interp_points=1000,rbnd=rbound,rwnd=rwindow)
jump2 = deepcopy(jump)

f = function (du,u,p,t)
  du[1] = 0.0 
end

prob = ODEProblem(f,[0.0],(0.0,10.0))
jump_prob = JumpProblem(prob,Extrande(),jump; rng=rng)

integrator = init(jump_prob,Tsit5())
sol = solve(jump_prob,Tsit5())

rate2 = (u,p,t) -> t < 5.0 ? 1.0 : 0.0
rbound2 = (u,p,t) -> 1.0 
jump3 = VariableRateJump(rate2,affect2!,interp_points=1000;rbnd=rbound2)

prob2 = ODEProblem(f,[0.0],(0.0,10.0))
jump_prob2 = JumpProblem(prob2,Extrande(),jump3; rng=rng)

sol2 = solve(jump_prob2,Tsit5())
@test sol2(5.0)[1] == sol2[end][1]
