using JumpProcesses, StochasticAD, Distributions, SciMLStructures
using Statistics, Random, Test

# Tests for BoundedSSA — a uniformization (thinning) SSA for jump-only
# ConstantRateJump/MassActionJump DiscreteProblems that composes with StochasticAD.
# Everything is exercised through the PUBLIC interface only:
#     solve(jprob, BoundedSSA(; rate_bound); saveat = ...)
# To differentiate, the JumpProblem is (re)built with the StochasticTriple parameter
# inside the estimator closure, then solved — no internal helpers are called.
# Runs in an isolated environment (GROUP=StochasticAD): needs only JumpProcesses +
# StochasticAD + Distributions (no ODE solver).

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
# parameter, not only plain `Vector`s. Only the `Tunable` portion is implemented.
struct TunableParams{T}
    tunables::T
end
SciMLStructures.isscimlstructure(::TunableParams) = true
function SciMLStructures.canonicalize(::SciMLStructures.Tunable, p::TunableParams)
    p.tunables, TunableParams, true
end

@testset "BoundedSSA StochasticAD (public solve interface)" begin

    # --- Test A: pure death, STATE-DEPENDENT rate (the case that matters) ------
    # rate = μ·u ; E[u(T)] = u0·e^{-μT} ; d/dμ E[u(T)] = -T·u0·e^{-μT}
    # A naive `while t<T` SSA returns 0 here; uniformization recovers the value.
    @testset "pure death (state-dependent)" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0     # Λ ≥ μ·u0 = 50, with margin
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        analytic = -T * u0 * exp(-μ0 * T)
        g, se = sad_partial([μ0], 1; N = 4000) do p
            jp = JumpProblem(DiscreteProblem([u0], (0.0, T), p), Direct(), death)
            solve(jp, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
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
        a, b = λ0 / μ0, exp(-μ0 * T)
        analytic = [(1 - b) / μ0, -λ0 / μ0^2 * (1 - b) + (u0 - a) * (-T * b)]
        for k in 1:2
            g, se = sad_partial([λ0, μ0], k; N = 4000) do p
                jp = JumpProblem(DiscreteProblem([u0], (0.0, T), p), Direct(), birth, death)
                solve(jp, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
            end
            @test abs(g - analytic[k]) < 4 * se
        end
    end

    # --- Test C: saveat returns the path at intermediate times (primal) --------
    # pure death: E[u(s)] = u0·e^{-μs} at every save time s.
    @testset "saveat path (intermediate times)" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ0]), Direct(), death)
        sat = [0.0, 0.5, 1.0]
        for (idx, s) in enumerate(sat)
            m, se = pmean(N = 4000) do
                solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = sat).u[idx][1]
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
            solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
        end
        ms, ses = pmean(N = 4000) do
            solve(jprob, SSAStepper()).u[end][1]
        end
        @test abs(mb - u0 * exp(-μ0 * T)) < 4 * seb
        @test abs(mb - ms) < 4 * sqrt(seb^2 + ses^2)      # agree within combined MC error
    end

    # --- Test E: solve interface — full path + piecewise-constant interpolation -
    @testset "solve returns a full path + interpolates" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ0]), Direct(), death)
        sol = solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = [0.0, 0.5, 1.0])
        @test sol.t == [0.0, 0.5, 1.0]
        @test length(sol.u) == 3
        @test sol.u[1][1] == u0                    # state at t0 is the initial condition
        @test 0 <= sol.u[end][1] <= u0
        @test sol(0.25)[1] == sol.u[1][1]          # piecewise-constant interpolation
    end

    # --- Test F: SciMLStructures parameter object (MTK/Catalyst tunables) -------
    # Same pure-death gradient as Test A, but the parameter is a structured
    # SciMLStructures object; BoundedSSA must differentiate through its tunable portion.
    @testset "SciMLStructures tunable parameters" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0
        death = ConstantRateJump((u, p, t) -> p.tunables[1] * u[1],
            integ -> (integ.u[1] -= 1; nothing))
        analytic = -T * u0 * exp(-μ0 * T)
        g, se = sad_partial([μ0], 1; N = 4000) do p
            jp = JumpProblem(DiscreteProblem([u0], (0.0, T), TunableParams(p)), Direct(), death)
            solve(jp, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
        end
        @test abs(g - analytic) < 4 * se
        @test abs(g) > 1.0
    end

    # --- Test G: MassActionJump, state-dependent rate, differentiated -----------
    # pure death X --> 0 at rate μ, as a MassActionJump with the rate constant taken
    # from p (param_idxs). Same analytic as Test A; exercises the MA propensity +
    # param_mapper differentiation path (no evalrxrate, no boolean on the population).
    @testset "MassActionJump pure death (differentiated)" begin
        T, u0, μ0, Λ = 1.0, 100, 0.5, 60.0
        majump = MassActionJump([[1 => 1]], [[1 => -1]]; param_idxs = [1])
        analytic = -T * u0 * exp(-μ0 * T)
        g, se = sad_partial([μ0], 1; N = 4000) do p
            jp = JumpProblem(DiscreteProblem([u0], (0.0, T), p), Direct(), majump)
            solve(jp, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
        end
        @test abs(g - analytic) < 4 * se
        @test abs(g) > 1.0
    end

    # --- Test H: order-2 MassActionJump, primal mean matches the stock SSA -------
    # dimerization 2X --> 0 at rate k; propensity = (k/2!)·u·(u-1). Validates the
    # falling-factorial propensity (order > 1) AND the combinatoric rate scaling against
    # SSAStepper/evalrxrate. max propensity ≈ (k/2)·u0·(u0-1) = 12.25 < Λ.
    @testset "MassActionJump dimerization (order 2, primal vs SSAStepper)" begin
        T, u0, k0, Λ = 1.0, 50, 0.01, 20.0
        majump = MassActionJump([[1 => 2]], [[1 => -2]]; param_idxs = [1])
        jprob = JumpProblem(DiscreteProblem([u0], (0.0, T), [k0]), Direct(), majump)
        mb, seb = pmean(N = 4000) do
            solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = [T]).u[end][1]
        end
        ms, ses = pmean(N = 4000) do
            solve(jprob, SSAStepper()).u[end][1]
        end
        @test abs(mb - ms) < 4 * sqrt(seb^2 + ses^2)
    end

    # --- guards: misuse should error, not silently mislead ---------------------
    @testset "guards" begin
        T = 1.0
        # missing rate_bound on the algorithm
        @test_throws ErrorException BoundedSSA()
        # non-additive (state-dependent) affect
        weird = ConstantRateJump((u, p, t) -> p[1], integ -> (integ.u[1] *= 2; nothing))
        jp_w = JumpProblem(DiscreteProblem([10], (0.0, T), [0.5]), Direct(), weird)
        @test_throws ErrorException solve(jp_w, BoundedSSA(; rate_bound = 10.0))
    end
end
