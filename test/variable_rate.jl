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

integrator = init(jump_prob,Tsit5())

sol = solve(jump_prob,Tsit5())

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

function ff(du,u,p,t)
    if p == 0
        du .= 1.01u
    else
        du .= 2.01u
    end
end

function gg(du,u,p,t)
  du[1,1] = 0.3u[1]
  du[1,2] = 0.6u[1]
  du[2,1] = 1.2u[1]
  du[2,2] = 0.2u[2]
end

rate_switch(u,p,t) = u[1]*1.0

function affect_switch!(integrator)
    integrator.p = 1
end

jump_switch = VariableRateJump(rate_switch,affect_switch!)

prob = SDEProblem(ff,gg,ones(2),(0.0,1.0),noise_rate_prototype=zeros(2,2))
jump_prob = JumpProblem(prob, Direct(), jump_switch)
solve(jump_prob, SRA1(), dt = 1.0)
