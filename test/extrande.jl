using DiffEqBase, JumpProcesses, OrdinaryDiffEq, Test
using StableRNGs
using Statistics
rng = StableRNG(48572)

f = function (du,u,p,t)
  du[1] = 0.0 
end

rate = (u,p,t) -> t < 5.0 ? 1.0 : 0.0
rbound = (u,p,t) -> 1.0 
rinterval = (u,p,t) -> Inf
affect! = (integrator) -> (integrator.u[1] = integrator.u[1]+1)
jump = VariableRateJump(rate,affect!;urate=rbound,rateinterval=rinterval)

prob = ODEProblem(f,[0.0],(0.0,10.0))
jump_prob = JumpProblem(prob,Extrande(),jump; rng=rng)

# Test that process doesn't jump when rate switches to 0. 
sol = solve(jump_prob,Tsit5())
@test sol(5.0)[1] == sol[end][1]

# Birth-death process with time-varying birth rates.
Nsims = 1000000
u0 = [10.0,]

function runsimulations(jump_prob, testts)
    Psamp = zeros(Int, length(testts), Nsims)
    for i in 1:Nsims
        sol_ = solve(jump_prob, Tsit5())
        Psamp[:, i] = getindex.(sol_(testts).u, 1)
    end
    mean(Psamp, dims=2)
end

# Variable rate birth jumps.
rateb = (u,p,t) -> (0.1*sin(t) + 0.2)
ratebbound = (u,p,t) -> 0.3
ratebwindow = (u,p,t) -> Inf
affectb! = (integrator) -> (integrator.u[1] = integrator.u[1] + 1)
jumpb = VariableRateJump(rateb, affectb!;urate=ratebbound, rateinterval=ratebwindow)

# Constant rate death jumps.
rated = (u,p,t) -> u[1] * 0.08
affectd! = (integrator) -> (integrator.u[1] = integrator.u[1] - 1)
jumpd = ConstantRateJump(rated, affectd!)

# Problem definition.
bd_prob = ODEProblem(f,u0,(0.0,2pi))
jump_bd_prob = JumpProblem(bd_prob, Extrande(), jumpb, jumpd) 

test_times = range(1.0, stop=2pi, length=3)
means = runsimulations(jump_bd_prob, test_times)

# ODE for the mean.
fu = function (du, u, p, t)
    du[1] = (0.1*sin(t) + 0.2) - (u[1] * 0.08)
end

ode_prob = ODEProblem(fu,u0,(0.0,2*pi))
ode_sol = solve(ode_prob, Tsit5())

# Test extrande against the ODE mean.
@test prod(isapprox.(means, getindex.(ode_sol(test_times).u, 1), rtol=1e-3))
