using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Test
using StableRNGs, Random

# ==========================================================================
# Test that rng/seed can be passed via solve/init kwargs for all pathways,
# and that JumpProblem(; rng=...) throws an error.
# ==========================================================================

# --------------------------------------------------------------------------
# Problem constructors
# --------------------------------------------------------------------------
function make_ssa_jump_prob()
    j1 = ConstantRateJump((u, p, t) -> 10.0, integrator -> (integrator.u[1] += 1))
    j2 = ConstantRateJump((u, p, t) -> 0.5 * u[1], integrator -> (integrator.u[1] -= 1))
    dprob = DiscreteProblem([10], (0.0, 20.0))
    JumpProblem(dprob, Direct(), j1, j2)
end

function make_ode_vr_jump_prob()
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    oprob = ODEProblem(f!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(oprob, Direct(), vrj)
end

function make_sde_vr_jump_prob()
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    g!(du, u, p, t) = (du[1] = 0.1 * u[1]; nothing)
    sprob = SDEProblem(f!, g!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(sprob, Direct(), vrj)
end

# ==========================================================================
# 1. JumpProblem(; rng=...) throws ArgumentError
# ==========================================================================
@testset "JumpProblem(; rng=...) throws ArgumentError" begin
    dprob = DiscreteProblem([10], (0.0, 10.0))
    j1 = ConstantRateJump((u, p, t) -> 1.0, integrator -> (integrator.u[1] += 1))
    @test_throws ArgumentError JumpProblem(dprob, Direct(), j1; rng = StableRNG(42))
end

# ==========================================================================
# 2. SSAStepper: rng via solve/init
# ==========================================================================
@testset "SSAStepper: rng via solve kwargs" begin
    jprob = make_ssa_jump_prob()
    integrator = init(jprob, SSAStepper(); rng = Xoshiro(42))
    @test SciMLBase.get_rng(integrator) isa Xoshiro
    sol = solve(jprob, SSAStepper(); rng = Xoshiro(42))
    @test sol.retcode == ReturnCode.Success
end

# ==========================================================================
# 3. SSAStepper: reproducibility via solve rng
# ==========================================================================
@testset "SSAStepper: solve rng reproducibility" begin
    jprob = make_ssa_jump_prob()
    sol1 = solve(jprob, SSAStepper(); rng = StableRNG(123))
    sol2 = solve(jprob, SSAStepper(); rng = StableRNG(123))
    @test sol1.t == sol2.t
    @test sol1.u == sol2.u
end

# ==========================================================================
# 4. SSAStepper: different seeds → different trajectories
# ==========================================================================
@testset "SSAStepper: different seeds → different trajectories" begin
    jprob = make_ssa_jump_prob()
    sol1 = solve(jprob, SSAStepper(); rng = StableRNG(100))
    sol2 = solve(jprob, SSAStepper(); rng = StableRNG(200))
    sol3 = solve(jprob, SSAStepper(); rng = StableRNG(300))
    times = [sol1.t[2], sol2.t[2], sol3.t[2]]
    @test allunique(times)
end

# ==========================================================================
# 5. ODE + VR: rng via solve/init
# ==========================================================================
@testset "ODE + VR: rng via solve kwargs" begin
    jprob = make_ode_vr_jump_prob()
    integrator = init(jprob, Tsit5(); rng = Xoshiro(42))
    @test SciMLBase.get_rng(integrator) isa Xoshiro
end

# ==========================================================================
# 6. ODE + VR: reproducibility via solve rng
# ==========================================================================
@testset "ODE + VR: solve rng reproducibility" begin
    jprob = make_ode_vr_jump_prob()
    sol1 = solve(jprob, Tsit5(); rng = StableRNG(123))
    sol2 = solve(jprob, Tsit5(); rng = StableRNG(123))
    @test sol1.t ≈ sol2.t
    @test sol1.u[end] ≈ sol2.u[end]
end

# ==========================================================================
# 7. ODE + VR: different seeds → different trajectories
# ==========================================================================
@testset "ODE + VR: different seeds → different trajectories" begin
    jprob = make_ode_vr_jump_prob()
    sols = [solve(jprob, Tsit5(); rng = StableRNG(s)) for s in (100, 200, 300)]
    finals = [s.u[end][1] for s in sols]
    @test allunique(finals)
end

# ==========================================================================
# 8. SDE + VR: rng via solve/init
# ==========================================================================
@testset "SDE + VR: rng via solve kwargs" begin
    jprob = make_sde_vr_jump_prob()
    integrator = init(jprob, EM(); dt = 0.01, rng = Xoshiro(42))
    @test SciMLBase.get_rng(integrator) isa Xoshiro
end

# ==========================================================================
# 9. SDE + VR: reproducibility via solve rng
# ==========================================================================
@testset "SDE + VR: solve rng reproducibility" begin
    jprob = make_sde_vr_jump_prob()
    sol1 = solve(jprob, EM(); dt = 0.01, save_everystep = false, rng = StableRNG(123))
    sol2 = solve(jprob, EM(); dt = 0.01, save_everystep = false, rng = StableRNG(123))
    @test sol1.u[end] ≈ sol2.u[end]
end

# ==========================================================================
# 10. SimpleTauLeaping: rng via solve kwargs
# ==========================================================================
@testset "SimpleTauLeaping: rng via solve kwargs" begin
    rate(out, u, p, t) = (out .= max.(u, 0); nothing)
    c(du, u, p, t, counts, mark) = (du .= counts; nothing)
    rj = RegularJump(rate, c, 2)
    dprob = DiscreteProblem([100, 100], (0.0, 1.0))
    jprob = JumpProblem(dprob, PureLeaping(), rj)
    sol1 = solve(jprob, SimpleTauLeaping(); dt = 0.01, rng = StableRNG(42))
    sol2 = solve(jprob, SimpleTauLeaping(); dt = 0.01, rng = StableRNG(42))
    @test sol1.u == sol2.u
end

# ==========================================================================
# 11. SimpleTauLeaping: different seeds → different trajectories
# ==========================================================================
@testset "SimpleTauLeaping: different seeds → different trajectories" begin
    rate(out, u, p, t) = (out .= max.(u, 0); nothing)
    c(du, u, p, t, counts, mark) = (du .= counts; nothing)
    rj = RegularJump(rate, c, 2)
    dprob = DiscreteProblem([100, 100], (0.0, 1.0))
    jprob = JumpProblem(dprob, PureLeaping(), rj)
    sol1 = solve(jprob, SimpleTauLeaping(); dt = 0.01, rng = StableRNG(42))
    sol2 = solve(jprob, SimpleTauLeaping(); dt = 0.01, rng = StableRNG(99))
    @test sol1.u != sol2.u
end

# ==========================================================================
# 12. has_rng / get_rng / set_rng! interface on SSAIntegrator
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
# 13. No rng kwarg: uses default_rng (non-reproducible but functional)
# ==========================================================================
@testset "No rng kwarg: functional solve" begin
    @testset "SSAStepper" begin
        jprob = make_ssa_jump_prob()
        sol = solve(jprob, SSAStepper())
        @test sol.retcode == ReturnCode.Success
    end

    @testset "ODE + VR" begin
        jprob = make_ode_vr_jump_prob()
        sol = solve(jprob, Tsit5())
        @test sol.retcode == ReturnCode.Success
    end
end

# ==========================================================================
# 14. seed kwarg: creates Xoshiro from integer seed
# ==========================================================================
@testset "seed kwarg creates Xoshiro" begin
    jprob = make_ssa_jump_prob()
    integrator = init(jprob, SSAStepper(); seed = 42)
    @test SciMLBase.get_rng(integrator) isa Xoshiro
end

# ==========================================================================
# 15. rng takes priority over seed
# ==========================================================================
@testset "rng takes priority over seed" begin
    jprob = make_ssa_jump_prob()
    integrator = init(jprob, SSAStepper(); rng = StableRNG(42), seed = 99)
    @test SciMLBase.get_rng(integrator) isa StableRNG
end
