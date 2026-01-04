using DiffEqBase
using JumpProcesses, OrdinaryDiffEq
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
solve(
    EnsembleProblem(jprob; safetycopy = true), SSAStepper(), EnsembleThreads();
    trajectories = 10
)

# test for https://github.com/SciML/JumpProcesses.jl/issues/472
let
    function f!(du, u, p, t)
        du[1] = -u[1]
        return nothing
    end
    u_0 = [1.0]
    ode_prob = ODEProblem(f!, u_0, (0.0, 10))
    vrj = VariableRateJump((u, p, t) -> 1.0, integrator -> nothing)

    for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jump_prob = JumpProblem(ode_prob, Direct(), vrj; vr_aggregator = agg)
        prob_func(prob, i, repeat) = deepcopy(prob)
        prob = EnsembleProblem(jump_prob, prob_func = prob_func)
        sol = solve(
            prob, Tsit5(), EnsembleThreads(), trajectories = 400,
            save_everystep = false
        )
        firstrx_time = [sol.u[i].t[2] for i in 1:length(sol)]
        @test allunique(firstrx_time)
    end
end
