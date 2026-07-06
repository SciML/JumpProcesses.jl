using JumpProcesses, StochasticAD, Distributions, SciMLStructures
using Statistics, Random, Test

# Tests for the optional StochasticAD extension: differentiating expectations over
# jump-only ConstantRateJump processes via BoundedSSA (uniformization/thinning).
# Needs only JumpProcesses + StochasticAD + Distributions (no ODE solver), so it
# runs in its own isolated environment (GROUP=StochasticAD).

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

# primal Monte-Carlo mean + standard error
function pmean(f; N)
    s = [f() for _ in 1:N]
    return mean(s), std(s) / sqrt(N)
end

# A minimal SciMLStructures parameter object standing in for an MTK/Catalyst
# `MTKParameters`: its differentiable tunables live in `.tunables`. Used to check that
# BoundedSSA seeds from and differentiates through the tunable portion of a *structured*
# parameter, not only plain `Vector`s. Only the `Tunable` portion is implemented — the
# sole portion BoundedSSA consults (via `_bssa_tunables`).
struct TunableParams{T}
    tunables::T
end
SciMLStructures.isscimlstructure(::TunableParams) = true
function SciMLStructures.canonicalize(::SciMLStructures.Tunable, p::TunableParams)
    p.tunables, TunableParams, true
end

@testset "StochasticAD constant-rate jumps (BoundedSSA, uniformization)" begin

    # --- Test A: pure death, STATE-DEPENDENT rate (the case that matters) ------
    # rate = μ·u ; E[u(T)] = u0·e^{-μT} ; d/dμ E[u(T)] = -T·u0·e^{-μT}
    # A naive `while t<T` SSA returns 0 here; uniformization recovers the value.
    @testset "pure death (state-dependent)" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0     # Λ ≥ μ·u0 = 50, with margin
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), death)
        analytic = -T * u0 * exp(-μ0 * T)
        g, se = sad_partial([μ0], 1; N = 10000) do p
            JumpProcesses.bounded_ssa_path(jprob, p; rate_bound = Λ, saveat = [T])[end][1]
        end
        @test abs(g - analytic) < 4 * se     # event-count derivative captured
        @test abs(g) > 1.0                    # explicitly NOT the zero a naive SSA gives
    end

    # --- Test B: birth-death, multi-channel + state-dependent ------------------
    # birth λ, death μ·u ; E[u(T)] = λ/μ + (u0-λ/μ)e^{-μT}
    @testset "birth-death (multi-channel, state-dependent)" begin
        T, u0, λ0, μ0, Λ = 1.0, 50, 10.0, 0.3, 60.0   # total = λ + μ·u ≤ ~28 ≪ Λ
        birth = ConstantRateJump((u, p, t) -> p[1],        integ -> (integ.u[1] += 1; nothing))
        death = ConstantRateJump((u, p, t) -> p[2] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), birth, death)
        a, b = λ0 / μ0, exp(-μ0 * T)
        analytic = [(1 - b) / μ0, -λ0 / μ0^2 * (1 - b) + (u0 - a) * (-T * b)]
        for k in 1:2
            g, se = sad_partial([λ0, μ0], k; N = 10000) do p
                JumpProcesses.bounded_ssa_path(jprob, p; rate_bound = Λ, saveat = [T])[end][1]
            end
            @test abs(g - analytic[k]) < 4 * se
        end
    end

    # --- Test C: saveat returns the path at intermediate times -----------------
    # pure death: E[u(s)] = u0·e^{-μs} at every save time s.
    @testset "saveat path (intermediate times)" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), death)
        sat = [0.0, 0.5, 1.0]
        for (idx, s) in enumerate(sat)
            m, se = pmean(N = 4000) do
                JumpProcesses.bounded_ssa_path(jprob, [μ0]; rate_bound = Λ, saveat = sat)[idx][1]
            end
            # `s = 0` is deterministic (u(0) = u0, se = 0); the `+ 1e-9` keeps the
            # zero-variance point from failing `0 < 0`.
            @test abs(m - u0 * exp(-μ0 * s)) <= 4 * se + 1e-9
        end
    end

    # --- Test D: primal mean matches the stock SSA -----------------------------
    @testset "primal mean matches stock SSAStepper" begin
        T, u0, μ0, Λ = 1.0, 100, 0.4, 60.0
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ0]), Direct(), death)
        mb, seb = pmean(N = 4000) do
            JumpProcesses.bounded_ssa_path(jprob, [μ0]; rate_bound = Λ, saveat = [T])[end][1]
        end
        ms, ses = pmean(N = 4000) do
            solve(jprob, SSAStepper()).u[end][1]
        end
        @test abs(mb - u0 * exp(-μ0 * T)) < 4 * seb
        @test abs(mb - ms) < 4 * sqrt(seb^2 + ses^2)      # agree within combined MC error
    end

    # --- Test E: BoundedSSA solve path (native interface) ----------------------
    @testset "BoundedSSA solve path + saveat" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        # primal: solve returns the full path at saveat
        jp0 = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ0]), Direct(), death)
        sol = solve(jp0, BoundedSSA(; rate_bound = Λ); saveat = [0.0, 0.5, 1.0])
        @test length(sol.u) == 3
        @test sol.u[1][1] == u0                # state at t0 is the initial condition
        @test 0 <= sol.u[end][1] <= u0
        # differentiate through solve (constructs jprob with triple parameters)
        analytic = -T * u0 * exp(-μ0 * T)
        g, se = sad_partial([μ0], 1; N = 4000) do p
            jp = JumpProblem(DiscreteProblem([u0], (0.0, T), p), Direct(), death)
            solve(jp, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
        end
        @test abs(g - analytic) < 4 * se
    end

    # --- Test F: SciMLStructures parameter object (MTK/Catalyst tunables) -------
    # Same pure-death gradient as Test A, but the parameter is a structured
    # SciMLStructures object; BoundedSSA must seed from and differentiate through its
    # tunable portion (a plain `Vector` canonicalizes to itself — covered by Test A).
    @testset "SciMLStructures tunable parameters" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0
        death = ConstantRateJump((u, p, t) -> p.tunables[1] * u[1],
            integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T)), Direct(), death)
        analytic = -T * u0 * exp(-μ0 * T)
        g, se = sad_partial([μ0], 1; N = 10000) do p
            JumpProcesses.bounded_ssa_path(jprob, TunableParams(p);
                rate_bound = Λ, saveat = [T])[end][1]
        end
        @test abs(g - analytic) < 4 * se     # tunable-portion derivative captured
        @test abs(g) > 1.0
    end

    # --- guards: misuse should error, not silently mislead ---------------------
    @testset "guards" begin
        T = 1.0
        # MassActionJump not yet supported
        majump = MassActionJump([0.5], [[1 => 1]], [[1 => -1]])
        jp_ma = JumpProblem(DiscreteProblem([10], (0.0, T), [0.5]), Direct(), majump)
        @test_throws ErrorException JumpProcesses.bounded_ssa_path(jp_ma, [0.5]; rate_bound = 10.0)
        # missing rate_bound on the algorithm
        @test_throws ErrorException BoundedSSA()
        # non-additive (state-dependent) affect
        weird = ConstantRateJump((u, p, t) -> p[1], integ -> (integ.u[1] *= 2; nothing))
        jp_w = JumpProblem(DiscreteProblem([10], (0.0, T)), Direct(), weird)
        @test_throws ErrorException JumpProcesses.bounded_ssa_path(jp_w, [0.5]; rate_bound = 10.0)
    end
end
