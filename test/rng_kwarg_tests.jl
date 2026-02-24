using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Test
using StableRNGs, Random

# ==========================================================================
# Test that rng can be passed via JumpProblem kwargs OR solve kwargs,
# and that solve-level rng takes precedence over JumpProblem-level rng.
#
# Strategy: use different RNG *types* to verify which one the integrator
# receives. StableRNG is passed at one level and Xoshiro at another,
# then we check the type on the integrator.
# ==========================================================================

# --------------------------------------------------------------------------
# Problem constructors
# --------------------------------------------------------------------------
function make_ssa_jump_prob(; kwargs...)
    j1 = ConstantRateJump((u, p, t) -> 10.0, integrator -> (integrator.u[1] += 1))
    j2 = ConstantRateJump((u, p, t) -> 0.5 * u[1], integrator -> (integrator.u[1] -= 1))
    dprob = DiscreteProblem([10], (0.0, 20.0))
    JumpProblem(dprob, Direct(), j1, j2; kwargs...)
end

function make_ode_vr_jump_prob(; kwargs...)
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    oprob = ODEProblem(f!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(oprob, Direct(), vrj; kwargs...)
end

function make_sde_vr_jump_prob(; kwargs...)
    f!(du, u, p, t) = (du[1] = -0.1 * u[1]; nothing)
    g!(du, u, p, t) = (du[1] = 0.1 * u[1]; nothing)
    sprob = SDEProblem(f!, g!, [100.0], (0.0, 10.0))
    vrj = VariableRateJump((u, p, t) -> 0.5 * u[1],
        integrator -> (integrator.u[1] -= 1.0))
    JumpProblem(sprob, Direct(), vrj; kwargs...)
end

# ==========================================================================
# 1. SSAStepper: rng via JumpProblem
# ==========================================================================
@testset "SSAStepper: rng via JumpProblem kwargs" begin
    xrng = Xoshiro(42)
    jprob = make_ssa_jump_prob(; rng = xrng)
    integrator = init(jprob, SSAStepper())
    @test SciMLBase.get_rng(integrator) isa Xoshiro
    sol = solve(jprob, SSAStepper())
    @test sol.retcode == ReturnCode.Success
end

# ==========================================================================
# 2. SSAStepper: rng via solve overrides JumpProblem
# ==========================================================================
@testset "SSAStepper: solve rng overrides JumpProblem rng" begin
    jprob = make_ssa_jump_prob(; rng = Xoshiro(42))
    integrator = init(jprob, SSAStepper(); rng = StableRNG(99))
    @test SciMLBase.get_rng(integrator) isa StableRNG
end

# ==========================================================================
# 3. SSAStepper: reproducibility via JumpProblem rng
# ==========================================================================
@testset "SSAStepper: JumpProblem rng reproducibility" begin
    jprob1 = make_ssa_jump_prob(; rng = StableRNG(123))
    jprob2 = make_ssa_jump_prob(; rng = StableRNG(123))
    sol1 = solve(jprob1, SSAStepper())
    sol2 = solve(jprob2, SSAStepper())
    @test sol1.t == sol2.t
    @test sol1.u == sol2.u
end

# ==========================================================================
# 4. SSAStepper: solve rng overrides for reproducibility
# ==========================================================================
@testset "SSAStepper: solve rng override reproducibility" begin
    jprob = make_ssa_jump_prob(; rng = Xoshiro(1))
    sol1 = solve(jprob, SSAStepper(); rng = StableRNG(42))
    sol2 = solve(jprob, SSAStepper(); rng = StableRNG(42))
    @test sol1.t == sol2.t
    @test sol1.u == sol2.u
end

# ==========================================================================
# 5. ODE + VR: rng via JumpProblem
# ==========================================================================
@testset "ODE + VR: rng via JumpProblem kwargs" begin
    jprob = make_ode_vr_jump_prob(; rng = Xoshiro(42))
    integrator = init(jprob, Tsit5())
    @test SciMLBase.get_rng(integrator) isa Xoshiro
end

# ==========================================================================
# 6. ODE + VR: solve rng overrides JumpProblem rng
# ==========================================================================
@testset "ODE + VR: solve rng overrides JumpProblem rng" begin
    jprob = make_ode_vr_jump_prob(; rng = Xoshiro(42))
    integrator = init(jprob, Tsit5(); rng = StableRNG(99))
    @test SciMLBase.get_rng(integrator) isa StableRNG
end

# ==========================================================================
# 7. ODE + VR: reproducibility via JumpProblem rng
# ==========================================================================
@testset "ODE + VR: JumpProblem rng reproducibility" begin
    jprob1 = make_ode_vr_jump_prob(; rng = StableRNG(123))
    jprob2 = make_ode_vr_jump_prob(; rng = StableRNG(123))
    sol1 = solve(jprob1, Tsit5())
    sol2 = solve(jprob2, Tsit5())
    @test sol1.t ≈ sol2.t
    @test sol1.u[end] ≈ sol2.u[end]
end

# ==========================================================================
# 8. ODE + VR: solve rng overrides for reproducibility
# ==========================================================================
@testset "ODE + VR: solve rng override reproducibility" begin
    jprob = make_ode_vr_jump_prob(; rng = Xoshiro(1))
    sol1 = solve(jprob, Tsit5(); rng = StableRNG(42))
    sol2 = solve(jprob, Tsit5(); rng = StableRNG(42))
    @test sol1.t ≈ sol2.t
    @test sol1.u[end] ≈ sol2.u[end]
end

# ==========================================================================
# 9. SDE + VR: rng via JumpProblem
# ==========================================================================
@testset "SDE + VR: rng via JumpProblem kwargs" begin
    jprob = make_sde_vr_jump_prob(; rng = Xoshiro(42))
    integrator = init(jprob, EM(); dt = 0.01)
    @test SciMLBase.get_rng(integrator) isa Xoshiro
end

# ==========================================================================
# 10. SDE + VR: solve rng overrides JumpProblem rng
# ==========================================================================
@testset "SDE + VR: solve rng overrides JumpProblem rng" begin
    jprob = make_sde_vr_jump_prob(; rng = Xoshiro(42))
    integrator = init(jprob, EM(); dt = 0.01, rng = StableRNG(99))
    @test SciMLBase.get_rng(integrator) isa StableRNG
end

# ==========================================================================
# 11. SDE + VR: reproducibility via JumpProblem rng
# ==========================================================================
@testset "SDE + VR: JumpProblem rng reproducibility" begin
    jprob1 = make_sde_vr_jump_prob(; rng = StableRNG(123))
    jprob2 = make_sde_vr_jump_prob(; rng = StableRNG(123))
    sol1 = solve(jprob1, EM(); dt = 0.01, save_everystep = false)
    sol2 = solve(jprob2, EM(); dt = 0.01, save_everystep = false)
    @test sol1.u[end] ≈ sol2.u[end]
end

# ==========================================================================
# 12. Tau-leaping: rng via JumpProblem
# ==========================================================================
@testset "SimpleTauLeaping: rng via JumpProblem kwargs" begin
    rate(out, u, p, t) = (out .= max.(u, 0); nothing)
    c(du, u, p, t, counts, mark) = (du .= counts; nothing)
    rj = RegularJump(rate, c, 2)
    dprob = DiscreteProblem([100, 100], (0.0, 1.0))
    jprob = JumpProblem(dprob, PureLeaping(), rj; rng = StableRNG(42))
    sol1 = solve(jprob, SimpleTauLeaping(); dt = 0.01)
    jprob2 = JumpProblem(dprob, PureLeaping(), rj; rng = StableRNG(42))
    sol2 = solve(jprob2, SimpleTauLeaping(); dt = 0.01)
    @test sol1.u == sol2.u
end

# ==========================================================================
# 13. Tau-leaping: solve rng overrides JumpProblem rng
# ==========================================================================
@testset "SimpleTauLeaping: solve rng overrides JumpProblem rng" begin
    rate(out, u, p, t) = (out .= max.(u, 0); nothing)
    c(du, u, p, t, counts, mark) = (du .= counts; nothing)
    rj = RegularJump(rate, c, 2)
    dprob = DiscreteProblem([100, 100], (0.0, 1.0))
    # JumpProblem has Xoshiro, solve has StableRNG
    jprob = JumpProblem(dprob, PureLeaping(), rj; rng = Xoshiro(1))
    sol1 = solve(jprob, SimpleTauLeaping(); dt = 0.01, rng = StableRNG(42))
    sol2 = solve(jprob, SimpleTauLeaping(); dt = 0.01, rng = StableRNG(42))
    @test sol1.u == sol2.u
    # Different from using the JumpProblem rng
    jprob2 = JumpProblem(dprob, PureLeaping(), rj; rng = Xoshiro(1))
    sol3 = solve(jprob2, SimpleTauLeaping(); dt = 0.01)
    @test sol1.u != sol3.u
end

# ==========================================================================
# 14. has_rng / get_rng / set_rng! interface on SSAIntegrator
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
end

# ==========================================================================
# 15. No rng kwarg: uses default_rng (non-reproducible but functional)
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
