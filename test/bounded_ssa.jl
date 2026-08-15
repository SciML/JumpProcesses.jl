# Ordinary (non-AD) tests for `BoundedSSA` — the uniformization/thinning SSA — exercised
# through the public `solve(jprob, BoundedSSA(; rate_bound); ...)` interface. Covers
# correctness against the stock `SSAStepper`, numeric-type genericity (`Float32` is not
# silently promoted to `Float64`), absorbing / zero-propensity safety, and RNG
# reproducibility via the `JumpProblem` rng. The StochasticAD differentiation tests live
# in the isolated `stochasticad/` environment (`test/stochasticad_tests.jl`).
using JumpProcesses, Statistics, Random, Test
using StableRNGs

@testset "BoundedSSA (ordinary SSA interface)" begin

    # --- linear reaction A --> B: known mean, matches SSAStepper -----------------
    # rate = k·A ; A(0) = A0 bounds the total propensity by k·A0. Each initial molecule
    # survives to T independently w.p. e^{-kT}, so A(T) ~ Binomial(A0, e^{-kT}) and the
    # standard error of an N-sample mean is known exactly. Compare BoundedSSA's Monte-Carlo
    # mean to the analytic mean and to the stock SSAStepper, at 5σ to keep CI non-flaky.
    @testset "linear reaction A->B vs SSAStepper" begin
        T, A0, k, N = 0.1, 100, 1.0, 4000
        Λ = k * A0                                     # exact global propensity bound
        psurv = exp(-k * T)
        expected = A0 * psurv
        se = sqrt(A0 * psurv * (1 - psurv) / N)        # exact SE of the mean (Binomial)
        rxn = ConstantRateJump((u, p, t) -> p[1] * u[1],
            integ -> (integ.u[1] -= 1; integ.u[2] += 1; nothing))
        jprob = JumpProblem(DiscreteProblem([A0, 0], (0.0, T), [k]), Direct(), rxn;
            rng = StableRNG(1234))
        mb = mean(solve(jprob, BoundedSSA(; rate_bound = Λ)).u[end][1] for _ in 1:N)
        ms = mean(solve(jprob, SSAStepper()).u[end][1] for _ in 1:N)
        @test abs(mb - expected) < 5 * se              # matches the analytic mean
        @test abs(mb - ms) < 5 * sqrt(2) * se          # matches the stock SSA
    end

    # --- Float32 state is preserved (no silent Float64 promotion) ----------------
    # Every value is Float32; assert no saved state is promoted to Float64.
    @testset "Float32 ConstantRateJump keeps Float32 state" begin
        T, Λ = 1.0f0, 60.0f0
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem(Float32[100], (0.0f0, T), Float32[0.5]),
            Direct(), death; rng = StableRNG(1))
        sol = solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = Float32[0.0, 0.5, 1.0])
        @test all(u -> eltype(u) === Float32, sol.u)   # no state promoted to Float64
        @test all(u -> all(isfinite, u), sol.u)
    end

    # --- Float32 MassActionJump path keeps Float32 state -------------------------
    # Protects the previously hard-coded Float64 delta allocation on the mass-action path.
    @testset "Float32 MassActionJump keeps Float32 state" begin
        T, Λ = 1.0f0, 60.0f0
        maj = MassActionJump([[1 => 1]], [[1 => -1]]; param_idxs = [1])
        jprob = JumpProblem(DiscreteProblem(Float32[100], (0.0f0, T), Float32[0.5]),
            Direct(), maj; rng = StableRNG(2))
        sol = solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = Float32[0.0, 0.5, 1.0])
        @test all(u -> eltype(u) === Float32, sol.u)
        @test all(u -> all(isfinite, u), sol.u)
    end

    # --- absorbing / zero-propensity state: no 0/0 in channel selection ----------
    # Regression test for the type-generic `tiny` sentinel that replaced `1e-300`: starting
    # already extinct with two death channels, every candidate event sees total propensity 0,
    # so channel selection evaluates `rand(Bernoulli(0 / (0 + tiny)))`. Without the sentinel
    # this is a 0/0 → NaN and `Bernoulli` throws; the absorbing state must be left unchanged
    # and no NaN/Inf may appear.
    @testset "zero-propensity state does not divide by zero" begin
        T, Λ = 1.0, 20.0
        u0 = [0]
        μ = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        ν = ConstantRateJump((u, p, t) -> p[2] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jprob = JumpProblem(DiscreteProblem(u0, (0.0, T), [1.0, 1.0]), Direct(), μ, ν;
            rng = StableRNG(7))
        sol = solve(jprob, BoundedSSA(; rate_bound = Λ); saveat = [0.0, 0.5, 1.0])
        @test sol.u[end] == u0                          # absorbing state is unchanged
        @test all(isfinite, sol.t)
        @test all(u -> all(isfinite, u), sol.u)         # would be NaN without the sentinel
    end

    # --- RNG reproducibility and independence from the global RNG ----------------
    # Compare whole saved trajectories (not just the final state, which two different runs
    # can share by chance) at closely spaced save times.
    @testset "reproducible from the JumpProblem rng" begin
        T, u0, μ, Λ = 1.0, 100, 0.5, 60.0
        sat = collect(0.0:0.1:T)
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        mkprob(seed) = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ]), Direct(), death;
            rng = StableRNG(seed))
        # identical seeds -> identical saved trajectories
        u1 = solve(mkprob(99), BoundedSSA(; rate_bound = Λ); saveat = sat).u
        u2 = solve(mkprob(99), BoundedSSA(; rate_bound = Λ); saveat = sat).u
        @test u1 == u2
        # a global reseed between two same-seed StableRNG solves must not change the trajectory
        jp1 = mkprob(99); Random.seed!(1)
        a = solve(jp1, BoundedSSA(; rate_bound = Λ); saveat = sat).u
        jp2 = mkprob(99); Random.seed!(2)
        b = solve(jp2, BoundedSSA(; rate_bound = Λ); saveat = sat).u
        @test a == b                                    # driven only by the problem rng
    end

    # --- solve(...; seed) reseeds the problem rng --------------------------------
    @testset "solve seed keyword reseeds the problem rng" begin
        T, u0, μ, Λ = 1.0, 100, 0.5, 60.0
        sat = collect(0.0:0.1:T)
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (integ.u[1] -= 1; nothing))
        jp = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ]), Direct(), death;
            rng = StableRNG(5))
        a = solve(jp, BoundedSSA(; rate_bound = Λ); seed = 12345, saveat = sat).u
        b = solve(jp, BoundedSSA(; rate_bound = Λ); seed = 12345, saveat = sat).u
        @test a == b
    end

    # --- scalar states are rejected early with a clear message -------------------
    @testset "scalar states are rejected" begin
        death = ConstantRateJump((u, p, t) -> 0.5, integ -> (integ.u -= 1; nothing))
        jp = JumpProblem(DiscreteProblem(10.0, (0.0, 1.0)), Direct(), death)
        @test_throws ArgumentError solve(jp, BoundedSSA(; rate_bound = 10.0))
        err = try
            solve(jp, BoundedSSA(; rate_bound = 10.0)); nothing
        catch e
            e
        end
        @test occursin("array-valued state", sprint(showerror, err))
    end

    # --- guards ------------------------------------------------------------------
    @testset "guards" begin
        @test_throws ArgumentError BoundedSSA()                     # missing rate_bound
        @test_throws ArgumentError BoundedSSA(; rate_bound = 0.0)   # non-positive bound
        # non-additive (state-dependent) affect is unsupported and must be rejected
        weird = ConstantRateJump((u, p, t) -> p[1], integ -> (integ.u[1] *= 2; nothing))
        jp_w = JumpProblem(DiscreteProblem([10], (0.0, 1.0), [0.5]), Direct(), weird)
        @test_throws ArgumentError solve(jp_w, BoundedSSA(; rate_bound = 10.0))
    end
end
