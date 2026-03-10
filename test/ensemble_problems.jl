using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Test
using StableRNGs, Random

# ==========================================================================
# Problem constructors
# ==========================================================================

# Constant-rate birth-death for SSAStepper / ODE-coupled tests
function make_ssa_jump_prob(; rng = StableRNG(12345))
    j1 = ConstantRateJump((u, p, t) -> 10.0, integrator -> (integrator.u[1] += 1))
    j2 = ConstantRateJump((u, p, t) -> 0.5 * u[1], integrator -> (integrator.u[1] -= 1))
    dprob = DiscreteProblem([10], (0.0, 20.0))
    JumpProblem(dprob, Direct(), j1, j2; rng)
end

# ODE + variable-rate jump
function make_vr_jump_prob(agg; rng = StableRNG(12345))
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    oprob = ODEProblem(f!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(oprob, Direct(), vrj; vr_aggregator = agg, rng)
end

# SDE + variable-rate jump
function make_sde_vr_jump_prob(agg; rng = StableRNG(12345))
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    g!(du, u, p, t) = (du[1] = 0.1 * u[1]; nothing)
    sprob = SDEProblem(f!, g!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(sprob, Direct(), vrj; vr_aggregator = agg, rng)
end

# Helpers
# First time strictly after t[1], robust to initialization saves at t=0.
first_jump_time(traj) = traj.t[findfirst(>(traj.t[1]), traj.t)]

# ==========================================================================
# 1. Serial ensemble: sequential trajectories get different RNG streams
# ==========================================================================

@testset "EnsembleSerial: distinct streams" begin
    @testset "SSAStepper" begin
        jprob = make_ssa_jump_prob()
        sol = solve(EnsembleProblem(jprob), SSAStepper(), EnsembleSerial();
            trajectories = 3)
        times = [first_jump_time(sol.u[i]) for i in 1:3]
        @test allunique(times)
    end

    @testset "ODE + VR ($agg)" for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = make_vr_jump_prob(agg)
        sol = solve(EnsembleProblem(jprob), Tsit5(), EnsembleSerial();
            trajectories = 3)
        times = [first_jump_time(sol.u[i]) for i in 1:3]
        @test allunique(times)
        finals = [sol.u[i].u[end][1] for i in 1:3]
        @test allunique(finals)
    end

    # EM() uses a fixed time grid so jump event times aren't directly visible
    # in t[2]; we check final values instead.
    @testset "SDE + VR (VR_FRM)" begin
        jprob = make_sde_vr_jump_prob(VR_FRM())
        sol = solve(EnsembleProblem(jprob), EM(), EnsembleSerial();
            trajectories = 3, dt = 0.01, save_everystep = false)
        finals = [sol.u[i].u[end][1] for i in 1:3]
        @test allunique(finals)
    end
end

# ==========================================================================
# 2. Sequential solves on same thread: RNG advances between solves
# ==========================================================================

@testset "Sequential solves: different RNG streams" begin
    @testset "SSAStepper" begin
        jprob = make_ssa_jump_prob()
        times = [first_jump_time(solve(jprob, SSAStepper())) for _ in 1:3]
        @test allunique(times)
    end

    @testset "ODE + VR ($agg)" for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = make_vr_jump_prob(agg)
        sols = [solve(jprob, Tsit5()) for _ in 1:3]
        times = [first_jump_time(s) for s in sols]
        @test allunique(times)
        finals = [s.u[end][1] for s in sols]
        @test allunique(finals)
    end
end

# ==========================================================================
# 3. Threaded ensemble: no data race on the shared JumpProblem
#
# The ODE/SSA path through __jump_init receives seed=nothing from
# SciMLBase, so deepcopy'd problems on non-main threads start with
# identical RNG states. We only assert completion here — uniqueness
# requires explicit seeding (tested in section 4 below).
#
# The SDE path goes through StochasticDiffEq's __init which generates
# per-trajectory seeds, so we can additionally verify uniqueness there.
# ==========================================================================

@testset "EnsembleThreads: no data race" begin
    @testset "SSAStepper" begin
        jprob = make_ssa_jump_prob()
        sol = solve(EnsembleProblem(jprob), SSAStepper(), EnsembleThreads();
            trajectories = 4)
        @test length(sol) == 4
    end

    @testset "ODE + VR ($agg)" for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = make_vr_jump_prob(agg)
        # This path previously had a data race: resetted_jump_problem called
        # randexp!(_jump_prob.rng, ...) on the shared original problem.
        sol = solve(EnsembleProblem(jprob), Tsit5(), EnsembleThreads();
            trajectories = 4, save_everystep = false)
        @test length(sol) == 4
    end

    @testset "SDE + VR (VR_FRM): unique trajectories" begin
        jprob = make_sde_vr_jump_prob(VR_FRM())
        # StochasticDiffEq generates per-trajectory seeds and passes them to
        # resetted_jump_problem, so trajectories should be distinct.
        sol = solve(EnsembleProblem(jprob), EM(), EnsembleThreads();
            trajectories = 4, dt = 0.01, save_everystep = false)
        @test length(sol) == 4
        finals = [sol.u[i].u[end][1] for i in 1:4]
        @test length(unique(finals)) > 1
    end
end

# ==========================================================================
# 4. Seed-based stream independence: resetted_jump_problem and
#    reset_jump_problem! produce distinct RNG streams for different seeds
#
# This tests the mechanism that EnsembleThreads relies on (when seeds are
# provided by the caller, e.g. StochasticDiffEq) to get independent streams
# on different threads.
# ==========================================================================

@testset "resetted_jump_problem: different seeds → different streams" begin
    jprob = make_ssa_jump_prob()
    seeds = UInt64[100, 200, 300]

    # Each seed should produce a distinct aggregator RNG state
    rngs = map(seeds) do s
        jp = JumpProcesses.resetted_jump_problem(jprob, s)
        jp.jump_callback.discrete_callbacks[1].condition.rng
    end
    draws = [rand(rng) for rng in rngs]
    @test allunique(draws)

    # Same seed should be deterministic
    jp1 = JumpProcesses.resetted_jump_problem(jprob, UInt64(42))
    jp2 = JumpProcesses.resetted_jump_problem(jprob, UInt64(42))
    rng1 = jp1.jump_callback.discrete_callbacks[1].condition.rng
    rng2 = jp2.jump_callback.discrete_callbacks[1].condition.rng
    @test rand(rng1) == rand(rng2)
end

@testset "reset_jump_problem!: different seeds → different streams" begin
    seeds = UInt64[100, 200, 300]
    draws = map(seeds) do s
        jp = make_ssa_jump_prob()
        JumpProcesses.reset_jump_problem!(jp, s)
        rand(jp.jump_callback.discrete_callbacks[1].condition.rng)
    end
    @test allunique(draws)
end

@testset "_derive_jump_seed: decorrelates from input seed" begin
    seed = UInt64(12345)
    derived = JumpProcesses._derive_jump_seed(seed)
    # Derived seed should differ from input
    @test derived != seed
    # Should be deterministic
    @test derived == JumpProcesses._derive_jump_seed(seed)
    # Different inputs → different outputs
    @test JumpProcesses._derive_jump_seed(UInt64(1)) != JumpProcesses._derive_jump_seed(UInt64(2))
end

# ==========================================================================
# 5. Variable-rate: jump_u thresholds are unique per trajectory
#
# For VR_FRM, each trajectory's first jump time is determined by the initial
# jump_u threshold (set to -randexp() by the VR_FRMEventCallback initialize).
# Distinct thresholds → distinct first event times.
# ==========================================================================

@testset "VR_FRM: jump_u thresholds unique per trajectory (EnsembleSerial)" begin
    jprob = make_vr_jump_prob(VR_FRM())
    sol = solve(EnsembleProblem(jprob), Tsit5(), EnsembleSerial();
        trajectories = 3)
    event_times = [first_jump_time(sol.u[i]) for i in 1:3]
    @test allunique(event_times)
end
