using JumpProcesses, DiffEqBase, OrdinaryDiffEq, StochasticDiffEq, Statistics
using Test
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> 1.0 * u[1]
affect! = function (integrator)
    integrator.u[1] = 1.0
end
jump1 = ConstantRateJump(rate, affect!)

prob = DiscreteProblem([10], (0.0, 50.0))
prob_control = DiscreteProblem([10], (0.0, 50.0))

jump_prob = JumpProblem(prob, Direct(), jump1; rng = rng)
jump_prob_control = JumpProblem(prob_control, Direct(), jump1; rng = rng)

coupling_map = [(1, 1)]
coupled_prob = SplitCoupledJumpProblem(jump_prob, jump_prob_control, Direct(), coupling_map;
    rng = rng)

@time sol = solve(coupled_prob, FunctionMap())
@time solve(jump_prob, FunctionMap())
@test [s[1] - s[2] for s in sol.u] == zeros(length(sol.t)) # coupling two copies of the same process should give zero

rate = (u, p, t) -> 1.0
affect! = function (integrator)
    integrator.u[1] = 1.0
end
jump1 = ConstantRateJump(rate, affect!)
rate = (u, p, t) -> 2.0
jump2 = ConstantRateJump(rate, affect!)

f = function (du, u, p, t)
    du[1] = u[1]
end
g = function (du, u, p, t)
    du[1] = 0.1
end

# Jump ODE to jump ODE
prob = ODEProblem(f, [1.0], (0.0, 1.0))
prob_control = ODEProblem(f, [1.0], (0.0, 1.0))
jump_prob = JumpProblem(prob, Direct(), jump1; rng = rng)
jump_prob_control = JumpProblem(prob_control, Direct(), jump2; rng = rng)
coupled_prob = SplitCoupledJumpProblem(jump_prob, jump_prob_control, Direct(), coupling_map;
    rng = rng)
sol = solve(coupled_prob, Tsit5())
@test mean([abs(s[1] - s[2]) for s in sol.u]) <= 5.0

# Jump SDE to Jump SDE
prob = SDEProblem(f, g, [1.0], (0.0, 1.0))
prob_control = SDEProblem(f, g, [1.0], (0.0, 1.0))
jump_prob = JumpProblem(prob, Direct(), jump1; rng = rng)
jump_prob_control = JumpProblem(prob_control, Direct(), jump1; rng = rng)
coupled_prob = SplitCoupledJumpProblem(jump_prob, jump_prob_control, Direct(), coupling_map;
    rng = rng)
sol = solve(coupled_prob, SRIW1())
@test mean([abs(s[1] - s[2]) for s in sol.u]) <= 5.0

# Jump SDE to Jump ODE
prob = ODEProblem(f, [1.0], (0.0, 1.0))
prob_control = SDEProblem(f, g, [1.0], (0.0, 1.0))
jump_prob = JumpProblem(prob, Direct(), jump1; rng = rng)
jump_prob_control = JumpProblem(prob_control, Direct(), jump1; rng = rng)
coupled_prob = SplitCoupledJumpProblem(jump_prob, jump_prob_control, Direct(), coupling_map;
    rng = rng)
sol = solve(coupled_prob, SRIW1())
@test mean([abs(s[1] - s[2]) for s in sol.u]) <= 5.0

# Jump SDE to Discrete
rate = (u, p, t) -> 1.0
affect! = function (integrator)
    integrator.u[1] += 1.0
end
prob = DiscreteProblem([1.0], (0.0, 1.0))
prob_control = SDEProblem(f, g, [1.0], (0.0, 1.0))
jump_prob = JumpProblem(prob, Direct(), jump1; rng = rng)
jump_prob_control = JumpProblem(prob_control, Direct(), jump1; rng = rng)
coupled_prob = SplitCoupledJumpProblem(jump_prob, jump_prob_control, Direct(), coupling_map;
    rng = rng)
sol = solve(coupled_prob, SRIW1())

# test mass action jumps coupled to ODE
# 0 -> A (stochasic) and A -> 0 (ODE)
rate = [100.0]
react_stoch = [Vector{Pair{Int, Int}}()]
net_stoch = [[1 => 1]]
majumps = MassActionJump(rate, react_stoch, net_stoch)
f = function (du, u, p, t)
    du[1] = -1.0 * u[1]
end
odeprob = ODEProblem(f, [10.0], (0.0, 10.0))
jump_prob = JumpProblem(odeprob, Direct(), majumps, save_positions = (false, false);
    rng = rng)
Nsims = 8000
Amean = 0.0
for i in 1:Nsims
    global Amean
    local sol = solve(jump_prob, Tsit5(), saveat = 10.0)
    Amean += sol[1, end]
end
Amean /= Nsims
actmean = 100.0 + (10.0 - 100.0) * exp(-1.0 * 10.0)
#println(abs(Amean-actmean)/actmean)
@test abs(actmean - Amean) < 0.02 * actmean
