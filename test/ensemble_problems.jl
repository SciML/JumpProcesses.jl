using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Test
using StableRNGs, Random

# ==========================================================================
# Problem constructors
# ==========================================================================

# Constant-rate birth-death for SSAStepper / ODE-coupled tests
function make_ssa_jump_prob()
    j1 = ConstantRateJump((u, p, t) -> 10.0, integrator -> (integrator.u[1] += 1))
    j2 = ConstantRateJump((u, p, t) -> 0.5 * u[1], integrator -> (integrator.u[1] -= 1))
    dprob = DiscreteProblem([10], (0.0, 20.0))
    JumpProblem(dprob, Direct(), j1, j2)
end

# ODE + variable-rate jump
function make_vr_jump_prob(agg)
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    oprob = ODEProblem(f!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(oprob, Direct(), vrj; vr_aggregator = agg)
end

# SDE + variable-rate jump
function make_sde_vr_jump_prob(agg)
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    g!(du, u, p, t) = (du[1] = 0.1 * u[1]; nothing)
    sprob = SDEProblem(f!, g!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(sprob, Direct(), vrj; vr_aggregator = agg)
end

# Helpers
first_jump_time(traj) = traj.t[2]

# ==========================================================================
# 1. Serial ensemble: sequential trajectories get different RNG streams
# ==========================================================================

@testset "EnsembleSerial: distinct streams" begin
    @testset "SSAStepper" begin
        jprob = make_ssa_jump_prob()
        sol = solve(EnsembleProblem(jprob), SSAStepper(), EnsembleSerial();
            trajectories = 3, rng = StableRNG(12345))
        times = [first_jump_time(sol.u[i]) for i in 1:3]
        @test allunique(times)
    end

    @testset "ODE + VR ($agg)" for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = make_vr_jump_prob(agg)
        sol = solve(EnsembleProblem(jprob), Tsit5(), EnsembleSerial();
            trajectories = 3, rng = StableRNG(12345))
        times = [first_jump_time(sol.u[i]) for i in 1:3]
        @test allunique(times)
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
        rng = StableRNG(12345)
        times = [first_jump_time(solve(jprob, SSAStepper(); rng)) for _ in 1:3]
        @test allunique(times)
    end

    @testset "ODE + VR ($agg)" for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = make_vr_jump_prob(agg)
        rng = StableRNG(12345)
        sols = [solve(jprob, Tsit5(); rng) for _ in 1:3]
        times = [first_jump_time(s) for s in sols]
        @test allunique(times)
    end
end

# ==========================================================================
# 3. Threaded ensemble: no data race on the shared JumpProblem
#
# With integrator-owned RNGs, each thread's integrator gets its own
# default_rng(). We only assert completion here — uniqueness is tested
# via explicit rng kwarg in section 4.
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
        sol = solve(EnsembleProblem(jprob), Tsit5(), EnsembleThreads();
            trajectories = 4, save_everystep = false)
        @test length(sol) == 4
    end

    @testset "SDE + VR (VR_FRM): unique trajectories" begin
        jprob = make_sde_vr_jump_prob(VR_FRM())
        # StochasticDiffEq generates per-trajectory seeds, so trajectories
        # should be distinct.
        sol = solve(EnsembleProblem(jprob), EM(), EnsembleThreads();
            trajectories = 4, dt = 0.01, save_everystep = false)
        @test length(sol) == 4
        finals = [sol.u[i].u[end][1] for i in 1:4]
        @test length(unique(finals)) > 1
    end
end

# ==========================================================================
# 4. rng kwarg reproducibility: same rng seed → identical trajectory,
#    different rng seeds → different trajectories
# ==========================================================================

@testset "rng kwarg reproducibility" begin
    @testset "SSAStepper: same seed → same trajectory" begin
        jprob = make_ssa_jump_prob()
        sol1 = solve(jprob, SSAStepper(); rng = StableRNG(42))
        sol2 = solve(jprob, SSAStepper(); rng = StableRNG(42))
        @test sol1.t == sol2.t
        @test sol1.u == sol2.u
    end

    @testset "SSAStepper: different seeds → different trajectories" begin
        jprob = make_ssa_jump_prob()
        sol1 = solve(jprob, SSAStepper(); rng = StableRNG(100))
        sol2 = solve(jprob, SSAStepper(); rng = StableRNG(200))
        sol3 = solve(jprob, SSAStepper(); rng = StableRNG(300))
        times = [first_jump_time(sol1), first_jump_time(sol2), first_jump_time(sol3)]
        @test allunique(times)
    end

    @testset "ODE + VR ($agg): same seed → same trajectory" for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = make_vr_jump_prob(agg)
        sol1 = solve(jprob, Tsit5(); rng = StableRNG(42))
        sol2 = solve(jprob, Tsit5(); rng = StableRNG(42))
        @test sol1.t ≈ sol2.t
        @test sol1.u[end] ≈ sol2.u[end]
    end

    @testset "ODE + VR ($agg): different seeds → different trajectories" for agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = make_vr_jump_prob(agg)
        sols = [solve(jprob, Tsit5(); rng = StableRNG(s)) for s in (100, 200, 300)]
        times = [first_jump_time(s) for s in sols]
        @test allunique(times)
    end
end

# ==========================================================================
# 5. has_rng / get_rng / set_rng! interface on SSAIntegrator
# ==========================================================================

@testset "SSAIntegrator RNG interface" begin
    jprob = make_ssa_jump_prob()
    integrator = init(jprob, SSAStepper(); rng = StableRNG(42))

    @test SciMLBase.has_rng(integrator)
    rng = SciMLBase.get_rng(integrator)
    @test rng isa StableRNG

    new_rng = StableRNG(99)
    SciMLBase.set_rng!(integrator, new_rng)
    @test SciMLBase.get_rng(integrator) === new_rng

    # mismatched RNG type should throw
    @test_throws ArgumentError SciMLBase.set_rng!(integrator, Random.Xoshiro(123))
end

# ==========================================================================
# 6. Variable-rate: jump_u thresholds are unique per trajectory
#
# For VR_FRM, each trajectory's first jump time is determined by the initial
# jump_u threshold (set to -randexp() by the VR_FRMEventCallback initialize).
# Distinct thresholds → distinct first event times, so we verify by checking
# that the second time point (first event) differs across serial trajectories.
# ==========================================================================

@testset "VR_FRM: jump_u thresholds unique per trajectory (EnsembleSerial)" begin
    jprob = make_vr_jump_prob(VR_FRM())
    sol = solve(EnsembleProblem(jprob), Tsit5(), EnsembleSerial();
        trajectories = 3, rng = StableRNG(12345))
    # The second time point is when the first variable-rate jump fires,
    # directly reflecting the initial -randexp() threshold.
    event_times = [sol.u[i].t[2] for i in 1:3]
    @test allunique(event_times)
end

# ==========================================================================
# 7. JumpProblem rng kwarg forwarded to solver
# ==========================================================================

@testset "JumpProblem rng kwarg throws ArgumentError" begin
    j1 = ConstantRateJump((u, p, t) -> 1.0, integrator -> (integrator.u[1] += 1))
    dprob = DiscreteProblem([10], (0.0, 10.0))
    j1_local = ConstantRateJump((u, p, t) -> 1.0, integrator -> (integrator.u[1] += 1))
    dprob_local = DiscreteProblem([10], (0.0, 10.0))
    jprob = JumpProblem(dprob_local, Direct(), j1_local)
    @test_throws ArgumentError JumpProblem(dprob_local, Direct(), j1_local; rng = StableRNG(1))
    sol = solve(jprob, SSAStepper(); rng = StableRNG(1))
    @test sol.retcode == ReturnCode.Success
end
