using DiffEqBase
using JumpProcesses, OrdinaryDiffEqTsit5 
using StableRNGs
rng = StableRNG(12345)

sr = [1.0, 2.0, 50.0]
maj = MassActionJump(sr, [[1 => 1], [1 => 1], [0 => 1]], [[1 => 1], [1 => -1], [1 => 1]])
params = (1.0, 2.0, 50.0)
tspan = (0.0, 4.0)
u0 = [5]
dprob = DiscreteProblem(u0, tspan, params)
jprob = JumpProblem(dprob, Direct(), maj; rng = rng)
solve(EnsembleProblem(jprob), SSAStepper(), EnsembleThreads(); trajectories = 10)
solve(EnsembleProblem(jprob; safetycopy = true), SSAStepper(), EnsembleThreads();
    trajectories = 10)

# test for https://github.com/SciML/JumpProcesses.jl/issues/472
let
    function f!(du, u, p, t)
        du[1] = -u[1]
        nothing 
    end
    u_0 = [1.0]
    ode_prob = ODEProblem(f!, u_0, (0.0, 10))
    rate(u, p, t) = 1.0
    jump!(integrator) = nothing
    jump_prob = JumpProblem(ode_prob, Direct(), VariableRateJump(rate, jump!); rng)
    prob_func(prob, i, repeat) = deepcopy(prob)
    prob = EnsembleProblem(jump_prob,prob_func = prob_func)
    solve(prob, Tsit5(), EnsembleThreads(), trajectories=10)
end