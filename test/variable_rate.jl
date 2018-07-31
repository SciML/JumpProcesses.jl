using DiffEqBase, DiffEqJump, OrdinaryDiffEq, StochasticDiffEq, Test

a = ExtendedJumpArray(rand(3),rand(2))
b = ExtendedJumpArray(rand(3),rand(2))

a.=b

@test a.u == b.u
@test a.jump_u == b.jump_u
@test a == b

c = rand(5)
d = 2.0

a .+ d
a .= b .+ d
a .+ c .+ d
a .= b .+ c .+ d

rate = (u,p,t) -> u[1]
affect! = (integrator) -> (integrator.u[1] = integrator.u[1]/2)
jump = VariableRateJump(rate,affect!,interp_points=1000)
jump2 = deepcopy(jump)

f = function (du,u,p,t)
  du[1] = u[1]
end

prob = ODEProblem(f,[0.2],(0.0,10.0))
jump_prob = JumpProblem(prob,Direct(),jump,jump2)

integrator = init(jump_prob,Tsit5(),dt=1/10)

sol = solve(jump_prob,Tsit5(),dt=1/10)

@test maximum([sol[i][2] for i in 1:length(sol)]) <= 1e-12
@test maximum([sol[i][3] for i in 1:length(sol)]) <= 1e-12

g = function (du,u,p,t)
  du[1] = u[1]
end

prob = SDEProblem(f,g,[0.2],(0.0,10.0))
jump_prob = JumpProblem(prob,Direct(),jump,jump2)

sol = solve(jump_prob,SRIW1())

@test maximum([sol[i][2] for i in 1:length(sol)]) <= 1e-12
@test maximum([sol[i][3] for i in 1:length(sol)]) <= 1e-12
