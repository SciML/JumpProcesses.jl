using JumpProcesses, StochasticAD, Distributions
using Statistics, Random, Test

# StochasticAD-compatible differentiation for jump-only ConstantRateJump SSA
# problems (the BoundedSSA path). Needs only JumpProcesses + StochasticAD +
# Distributions (no ODE solver), so it runs in its own isolated environment
# (GROUP=StochasticAD).

# per-partial StochasticAD gradient with fixed seeds (reproducible)
function sad_partial(f, p0, k; N)
    s = Vector{Float64}(undef, N)
    for i in 1:N
        Random.seed!(i)
        s[i] = derivative_estimate(p0[k]) do pk
            p = [j == k ? pk : oftype(pk, p0[j]) for j in eachindex(p0)]
            f(p)
        end
    end
    return mean(s), std(s) / sqrt(N)
end

@testset "StochasticAD constant-rate jumps (BoundedSSA)" begin

    # --- Test A: pure death, STATE-DEPENDENT rate (the case that matters) ---
    # rate = μ·u[1] ; E[u(T)] = u0·e^{-μT} ; d/dμ E[u(T)] = -T·u0·e^{-μT}
    # A naive `while t<T` SSA returns derivative 0 here; the fixed-length
    # Bernoulli-per-event reformulation must recover the analytic value.
    @testset "pure death (state-dependent)" begin
        T, u0, μ0 = 1.0, 100, 0.5
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), death)
        analytic = -T * u0 * exp(-μ0 * T)
        g, se = sad_partial([μ0], 1; N = 10000) do p
            bounded_ssa_final_state(jprob, p; nmax = 200)[1]
        end
        @test abs(g - analytic) < 4 * se     # event-count derivative captured
        @test abs(g) > 1.0                    # explicitly NOT the zero a naive SSA gives
    end

    # --- Test B: birth-death, multi-channel + state-dependent ---
    # birth λ, death μ·u[1] ; E[u(T)] = λ/μ + (u0-λ/μ)e^{-μT}
    @testset "birth-death (multi-channel, state-dependent)" begin
        T, u0, λ0, μ0 = 1.0, 50, 10.0, 0.3
        birth = ConstantRateJump((u, p, t) -> p[1],        integ -> (integ.u[1] += 1; nothing))
        death = ConstantRateJump((u, p, t) -> p[2] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), birth, death)
        a, b = λ0 / μ0, exp(-μ0 * T)
        analytic = [(1 - b) / μ0, -λ0 / μ0^2 * (1 - b) + (u0 - a) * (-T * b)]
        for k in 1:2
            g, se = sad_partial([λ0, μ0], k; N = 10000) do p
                bounded_ssa_final_state(jprob, p; nmax = 400)[1]
            end
            @test abs(g - analytic[k]) < 4 * se
        end
    end

    # --- Test C: homogeneous Poisson baseline (state-INDEPENDENT) ---
    # two constant channels ; u(T) = u0 + N1 - N2 ; d/dλ1 = T, d/dλ2 = -T
    @testset "homogeneous Poisson baseline" begin
        T, u0 = 2.0, 10
        j1 = ConstantRateJump((u, p, t) -> p[1], integ -> (integ.u[1] += 1; nothing))
        j2 = ConstantRateJump((u, p, t) -> p[2], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), j1, j2)
        λ0 = [3.0, 1.0]
        for k in 1:2
            g, se = sad_partial(λ0, k; N = 10000) do p
                bounded_ssa_final_state(jprob, p; nmax = 200)[1]
            end
            @test abs(g - (k == 1 ? T : -T)) < 4 * se
        end
    end

    # --- Test D: BoundedSSA solve path differentiates (native API) ---
    # solve(jprob, BoundedSSA(; nmax)) must give the same gradient as the core.
    # Constructs the JumpProblem with triple parameters, exercising the generic
    # rate cache during construction and the BoundedSSA __solve.
    @testset "BoundedSSA solve path" begin
        T, u0, μ0 = 1.0, 100, 0.5
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        analytic = -T * u0 * exp(-μ0 * T)
        # primal: solve returns a sensible final state
        jp0 = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ0]), Direct(), death)
        sol = solve(jp0, BoundedSSA(; nmax = 200))
        @test length(sol.u[end]) == 1
        @test 0 <= sol.u[end][1] <= u0
        # differentiate through solve
        g, se = sad_partial([μ0], 1; N = 4000) do p
            jp = JumpProblem(DiscreteProblem([u0], (0.0, T), p), Direct(), death)
            solve(jp, BoundedSSA(; nmax = 200)).u[end][1]
        end
        @test abs(g - analytic) < 4 * se
    end

    # --- Test E: primal distribution matches the stock SSA ---
    # at large nmax the bounded SSA's mean must match solve(jprob, SSAStepper()).
    @testset "primal mean matches stock SSAStepper" begin
        T, u0, μ0 = 1.0, 100, 0.4
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ0]), Direct(), death)
        nT = 4000
        bounded = Vector{Float64}(undef, nT)
        stock = Vector{Float64}(undef, nT)
        for i in 1:nT
            Random.seed!(i)
            bounded[i] = bounded_ssa_final_state(jprob, [μ0]; nmax = 400)[1]
            Random.seed!(10_000 + i)
            stock[i] = solve(jprob, SSAStepper()).u[end][1]
        end
        analytic = u0 * exp(-μ0 * T)
        @test abs(mean(bounded) - analytic) < 4 * std(bounded) / sqrt(nT)
        # bounded vs stock means agree within combined MC error
        se = sqrt(var(bounded) / nT + var(stock) / nT)
        @test abs(mean(bounded) - mean(stock)) < 4 * se
    end

    # --- Test F: saturation diagnostic ---
    @testset "saturation_probability" begin
        T, u0, λ0 = 1.0, 0, 20.0
        birth = ConstantRateJump((u, p, t) -> p[1], integ -> (integ.u[1] += 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), birth)
        # ~Poisson(20) events: nmax=5 is far too small, nmax=200 is ample
        @test saturation_probability(jprob, [λ0]; nmax = 5, ntrials = 2000) > 0.5
        @test saturation_probability(jprob, [λ0]; nmax = 200, ntrials = 2000) < 0.01
    end

    # --- guards: misuse should error, not silently mislead ---
    @testset "guards" begin
        T = 1.0
        # MassActionJump not yet supported
        majump = MassActionJump([0.5], [[1 => 1]], [[1 => -1]])
        jp_ma = JumpProblem(DiscreteProblem([10], (0.0, T), [0.5]), Direct(), majump)
        @test_throws ErrorException bounded_ssa_final_state(jp_ma, [0.5]; nmax = 50)
        # missing nmax on the algorithm
        @test_throws ErrorException BoundedSSA()
        # non-additive (state-dependent) affect
        weird = ConstantRateJump((u, p, t) -> p[1], integ -> (integ.u[1] *= 2; nothing))
        jp_w = JumpProblem(DiscreteProblem([10], (0.0, T)), Direct(), weird)
        @test_throws ErrorException bounded_ssa_final_state(jp_w, [0.5]; nmax = 50)
    end

    # --- generic rate cache (Chris's literal request): a triple-valued rate must
    # pass through the existing Direct rate-cache without Float64(::StochasticTriple).
    # SCOPE: this only makes the cache generic. It does NOT by itself give correct
    # gradients — the stock `while t < T` event boundary still drops the event-count
    # derivative, and a full stock solve would next hit the SSAStepper integrator's
    # Float64 time. Correct gradients use BoundedSSA / bounded_ssa_final_state.
    @testset "generic rate cache (no Float64(::StochasticTriple))" begin
        st = stochastic_triple(identity, 0.5)
        jump = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        rates, affects = JumpProcesses.get_jump_info_tuples((jump,))
        agg = JumpProcesses.build_jump_aggregation(
            JumpProcesses.DirectJumpAggregation, [10], [st], 0.0, 1.0, nothing,
            rates, affects, (false, false), JumpProcesses.DEFAULT_RNG)
        @test eltype(agg.cur_rates) <: StochasticAD.StochasticTriple   # cache is generic, not Float64
        sr, _ = JumpProcesses.time_to_next_jump(agg, [10], [st], 0.0)
        @test sr isa StochasticAD.StochasticTriple                      # filled without Float64 error
    end
end
