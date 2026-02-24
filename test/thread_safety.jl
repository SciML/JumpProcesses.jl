using DiffEqBase, Test
using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq

sr = [1.0, 2.0, 50.0]
maj = MassActionJump(sr, [[1 => 1], [1 => 1], [0 => 1]], [[1 => 1], [1 => -1], [1 => 1]])
params = (1.0, 2.0, 50.0)
tspan = (0.0, 4.0)
u0 = [5]
dprob = DiscreteProblem(u0, tspan, params)
jprob = JumpProblem(dprob, Direct(), maj)

# Verify threaded solves complete and produce distinct trajectories.
# NOTE: We intentionally do NOT pass `rng` here. In threaded ensembles, passing a
# shared rng object via `solve(...; rng=...)` does not yet provide correct
# per-trajectory stream handling. Until SciMLBase's ensemble RNG updates land
# (master rng -> per-trajectory rng), correctness in threaded contexts relies on
# task-local `Random.default_rng()`.
sol = solve(EnsembleProblem(jprob), SSAStepper(), EnsembleThreads();
    trajectories = 400)
@test length(sol) == 400
firstrx_time = [sol.u[i].t[findfirst(>(sol.u[i].t[1]), sol.u[i].t)] for i in 1:length(sol)]
@test allunique(firstrx_time)

sol2 = solve(EnsembleProblem(jprob; safetycopy = true), SSAStepper(), EnsembleThreads();
    trajectories = 400)
@test length(sol2) == 400
firstrx_time2 = [sol2.u[i].t[findfirst(>(sol2.u[i].t[1]), sol2.u[i].t)] for i in 1:length(sol2)]
@test allunique(firstrx_time2)

# test for https://github.com/SciML/JumpProcesses.jl/issues/472
let
    function f!(du, u, p, t)
        du[1] = -u[1]
        nothing
    end
    u_0 = [1.0]
    ode_prob = ODEProblem(f!, u_0, (0.0, 10))
    vrj = VariableRateJump((u, p, t) -> 1.0, integrator -> nothing)

    for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jump_prob = JumpProblem(ode_prob, Direct(), vrj; vr_aggregator = agg)
        prob_func(prob, i, repeat) = deepcopy(prob)
        prob = EnsembleProblem(jump_prob, prob_func = prob_func)
        sol = solve(prob, Tsit5(), EnsembleThreads(), trajectories = 400,
            save_everystep = false)
        firstrx_time = [sol.u[i].t[findfirst(>(sol.u[i].t[1]), sol.u[i].t)] for i in 1:length(sol)]
        @test allunique(firstrx_time)
    end
end

# SDE + variable-rate jumps with EnsembleThreads
let
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    g!(du, u, p, t) = (du[1] = 0.1 * u[1]; nothing)
    sde_prob = SDEProblem(f!, g!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))

    for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jump_prob = JumpProblem(sde_prob, Direct(), vrj; vr_aggregator = agg)
        prob_func(prob, i, repeat) = deepcopy(prob)
        prob = EnsembleProblem(jump_prob; prob_func)
        sol = solve(prob, SRIW1(), EnsembleThreads();
            trajectories = 400, save_everystep = false)
        firstrx_time = [sol.u[i].t[findfirst(>(sol.u[i].t[1]), sol.u[i].t)] for i in 1:length(sol)]
        @test allunique(firstrx_time)
    end
end
