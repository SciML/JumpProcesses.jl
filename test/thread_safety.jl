using DiffEqBase, SciMLBase, Test
using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq
using StableRNGs

sr = [1.0, 2.0, 50.0]
maj = MassActionJump(sr, [[1 => 1], [1 => 1], [0 => 1]], [[1 => 1], [1 => -1], [1 => 1]])
params = (1.0, 2.0, 50.0)
tspan = (0.0, 4.0)
u0 = [5]
dprob = DiscreteProblem(u0, tspan, params)
jprob = JumpProblem(dprob, Direct(), maj)

# Keep coverage of default task-local RNGs as well as the explicit ensemble RNG
# controls below. SciMLBase uses an ensemble `rng` or `seed` to assign streams to
# individual trajectories; the master RNG is not shared by the jump integrators.
sol = solve(EnsembleProblem(jprob), SSAStepper(), EnsembleThreads();
    trajectories = 400)
@test length(sol.u) == 400
firstrx_time = [sol.u[i].t[findfirst(>(sol.u[i].t[1]), sol.u[i].t)]
                for i in 1:length(sol.u)]
@test allunique(firstrx_time)

sol2 = solve(EnsembleProblem(jprob; safetycopy = true), SSAStepper(), EnsembleThreads();
    trajectories = 400)
@test length(sol2.u) == 400
firstrx_time2 = [sol2.u[i].t[findfirst(>(sol2.u[i].t[1]), sol2.u[i].t)]
                 for i in 1:length(sol2.u)]
@test allunique(firstrx_time2)

@testset "Threaded ensemble RNG and seed controls" begin
    paths(sol) = [(trajectory.t, trajectory.u) for trajectory in sol.u]
    first_event_times(sol) = [trajectory.t[findfirst(>(first(trajectory.t)), trajectory.t)]
                              for trajectory in sol.u]
    for safetycopy in (false, true), input in (:rng, :seed)

        @testset "safetycopy=$safetycopy, $input" begin
            # Construct a fresh master RNG for each ensemble solve. Only the
            # ensemble layer uses this object; each trajectory receives its own RNG.
            rng_options(seed) = input === :rng ? (; rng = StableRNG(seed)) : (; seed)
            ensemble = EnsembleProblem(jprob; safetycopy)
            serial = solve(ensemble, SSAStepper(), EnsembleSerial();
                trajectories = 16, rng_options(12345)...)
            threaded = solve(ensemble, SSAStepper(), EnsembleThreads();
                trajectories = 16, rng_options(12345)...)
            replay = solve(ensemble, SSAStepper(), EnsembleThreads();
                trajectories = 16, rng_options(12345)...)
            changed = solve(ensemble, SSAStepper(), EnsembleThreads();
                trajectories = 16, rng_options(54321)...)

            @test length(threaded.u) == 16
            @test all(SciMLBase.successful_retcode, threaded.u)
            @test paths(threaded) == paths(serial)
            @test paths(threaded) == paths(replay)
            @test first_event_times(threaded) != first_event_times(changed)
            @test allunique(first_event_times(threaded))
        end
    end
end

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
        prob = EnsembleProblem(jump_prob)
        sol = solve(prob, Tsit5(), EnsembleThreads(), trajectories = 400,
            save_everystep = false)
        firstrx_time = [sol.u[i].t[findfirst(>(sol.u[i].t[1]), sol.u[i].t)]
                        for i in 1:length(sol.u)]
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
        prob = EnsembleProblem(jump_prob)
        sol = solve(prob, SRIW1(), EnsembleThreads();
            trajectories = 400, save_everystep = false)
        firstrx_time = [sol.u[i].t[findfirst(>(sol.u[i].t[1]), sol.u[i].t)]
                        for i in 1:length(sol.u)]
        @test allunique(firstrx_time)
    end
end
