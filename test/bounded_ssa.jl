# Ordinary (non-AD) tests for `BoundedSSA` — the uniformization/thinning SSA — exercised
# through the public `solve(jprob, BoundedSSA(; rate_bound); ...)` interface. Covers
# correctness against the stock `SSAStepper`, numeric-type genericity (`Float32` is not
# silently promoted to `Float64`), absorbing / zero-propensity safety, and RNG
# reproducibility via the `JumpProblem` rng. The StochasticAD differentiation tests live
# in the isolated `stochasticad/` environment (`test/stochasticad_tests.jl`).
using JumpProcesses, Statistics, Random, Test
using StableRNGs
using SciMLBase: DiscreteProblem

# Minimal SciMLStructures parameter object (a stand-in for an MTK/Catalyst `MTKParameters`)
# whose tunables live in `.tunables`, used to exercise the `_bssa_tunables` / seeding path on
# a *structured* parameter in the ordinary (non-AD) solve. Reached through
# `JumpProcesses.SciMLStructures` so this test needs no extra dependency.
const _SS = JumpProcesses.SciMLStructures
struct BSSATunable{T}
    tunables::T
end

_SS.isscimlstructure(::BSSATunable) = true
_SS.hasportion(::_SS.Tunable, ::BSSATunable) = true
_SS.canonicalize(::_SS.Tunable, p::BSSATunable) = (p.tunables, BSSATunable, true)

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
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (
            integ.u[1] -= 1; nothing))
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
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (
            integ.u[1] -= 1; nothing))
        mkprob(seed) = JumpProblem(DiscreteProblem([u0], (0.0, T), [μ]), Direct(), death;
            rng = StableRNG(seed))
        # identical seeds -> identical saved trajectories
        u1 = solve(mkprob(99), BoundedSSA(; rate_bound = Λ); saveat = sat).u
        u2 = solve(mkprob(99), BoundedSSA(; rate_bound = Λ); saveat = sat).u
        @test u1 == u2
        # a global reseed between two same-seed StableRNG solves must not change the trajectory
        jp1 = mkprob(99);
        Random.seed!(1)
        a = solve(jp1, BoundedSSA(; rate_bound = Λ); saveat = sat).u
        jp2 = mkprob(99);
        Random.seed!(2)
        b = solve(jp2, BoundedSSA(; rate_bound = Λ); saveat = sat).u
        @test a == b                                    # driven only by the problem rng
    end

    # --- solve(...; seed) reseeds the problem rng --------------------------------
    @testset "solve seed keyword reseeds the problem rng" begin
        T, u0, μ, Λ = 1.0, 100, 0.5, 60.0
        sat = collect(0.0:0.1:T)
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (
            integ.u[1] -= 1; nothing))
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
            solve(jp, BoundedSSA(; rate_bound = 10.0));
            nothing
        catch e
            e
        end
        @test occursin("array-valued state", sprint(showerror, err))
    end

    # --- guards ------------------------------------------------------------------
    @testset "guards" begin
        @test_throws ArgumentError BoundedSSA()                     # missing rate_bound
        @test_throws ArgumentError BoundedSSA(; rate_bound = 0.0)   # non-positive bound
    end

    # --- non-additive ConstantRateJump affects are rejected, not simulated incorrectly ----
    # BoundedSSA infers a constant additive net change Δ from `affect!` by probing it at
    # two states. State-dependent affects that produce different Δ values are rejected.
    #
    # Time- and parameter-dependent affects are also unsupported, but are not necessarily
    # detectable by this inference, so we deliberately do not assert that they throw.
    @testset "non-additive affects are rejected" begin
        # Self state-dependence: Δ = u[1].
        doubling = ConstantRateJump(
            (u, p, t) -> p[1],
            integ -> (integ.u[1] *= 2; nothing)
        )

        jp1 = JumpProblem(
            DiscreteProblem([10], (0.0, 1.0), [0.5]),
            Direct(),
            doubling
        )

        err = try
            solve(jp1, BoundedSSA(; rate_bound = 10.0))
            nothing
        catch e
            e
        end

        @test err isa ArgumentError
        @test err isa ArgumentError &&
              occursin("additive", sprint(showerror, err))

        # Cross-component state-dependence:
        # Δu[1] depends on the current value of u[2].
        coupled = ConstantRateJump(
            (u, p, t) -> p[1],
            integ -> (integ.u[1] += integ.u[2]; nothing)
        )

        jp2 = JumpProblem(
            DiscreteProblem([10, 5], (0.0, 1.0), [0.5]),
            Direct(),
            coupled
        )

        @test_throws ArgumentError solve(
            jp2,
            BoundedSSA(; rate_bound = 10.0)
        )
    end

    # Parameter container forms: the parameter extraction / state-seeding path must not
    # assume that `p` is a mutable Vector. Each ordinary solve exercises `_bssa_tunables(p)`
    # and the `z = 0 * sum(...)` seed for one supported container form. This is API/type
    # coverage, not another statistical validation, so a single seeded solve per form
    # suffices: we assert the pure-death invariants rather than a Monte-Carlo mean.
    # StochasticAD differentiation coverage lives in stochasticad_tests.jl.
    @testset "parameter container forms" begin
        T, u0 = 0.5, 100
        sat = [0.0, T]
        # Solve once, reproducibly, and check the container form solves and preserves the
        # pure-death invariants. `Λ` here is only the uniformization bound — it is never used
        # to derive the model's physics, so a loose bound must give the same invariants as a
        # tight one.
        function checksolve(jprob, Λ)
            sol = solve(jprob, BoundedSSA(; rate_bound = Λ); seed = 1, saveat = sat)
            @test sol.u[1][1] == u0                 # state at t0 is the initial condition
            @test all(isfinite, sol.u[end])
            @test 0 <= sol.u[end][1] <= u0          # pure death stays within [0, u0]
        end

        # NullParameters
        death0 = ConstantRateJump(
            (u, p, t) -> u[1],
            integ -> (integ.u[1] -= 1; nothing)
        )
        jp0 = JumpProblem(
            DiscreteProblem([u0], (0.0, T)),
            Direct(),
            death0;
            rng = StableRNG(1)
        )
        checksolve(jp0, Float64(u0))

        # Scalar
        deaths = ConstantRateJump(
            (u, p, t) -> p * u[1],
            integ -> (integ.u[1] -= 1; nothing)
        )
        jps = JumpProblem(
            DiscreteProblem([u0], (0.0, T), 0.5),
            Direct(),
            deaths;
            rng = StableRNG(2)
        )
        checksolve(jps, 0.5 * u0)

        # Tuple
        deatht = ConstantRateJump(
            (u, p, t) -> p[1] * u[1],
            integ -> (integ.u[1] -= 1; nothing)
        )
        jpt = JumpProblem(
            DiscreteProblem([u0], (0.0, T), (0.5,)),
            Direct(),
            deatht;
            rng = StableRNG(3)
        )
        checksolve(jpt, 0.5 * u0)

        # SciMLStructures
        deathx = ConstantRateJump(
            (u, p, t) -> p.tunables[1] * u[1],
            integ -> (integ.u[1] -= 1; nothing)
        )
        jpx = JumpProblem(
            DiscreteProblem(
                [u0],
                (0.0, T),
                BSSATunable([0.5])
            ),
            Direct(),
            deathx;
            rng = StableRNG(4)
        )
        checksolve(jpx, 0.5 * u0)
    end

    # --- a too-small rate_bound is reported with a BoundedSSA-specific error -------
    # If Λ underbounds the total propensity, uniformization would need an accept probability
    # total/Λ > 1. BoundedSSA detects this at the offending candidate event and throws a
    # clear ArgumentError naming the total, the bound and the time, rather than letting
    # `Bernoulli` fail with a generic domain error — and it does NOT clamp the probability
    # (which would silently change the sampled process).
    @testset "rate_bound violation is reported clearly" begin
        # rate = u[1] with u0 = 100 gives total propensity 100 at t0, but Λ = 50 underbounds
        # it. Uniformization first draws candidate times ~ Poisson(Λ·ΔT); under the fixed
        # StableRNG below there is at least one candidate, and the first candidate — evaluated
        # at u0 — already has total = 100 > Λ, so the violation is reported reproducibly.
        death = ConstantRateJump((u, p, t) -> p[1] * u[1], integ -> (
            integ.u[1] -= 1; nothing))
        jp = JumpProblem(DiscreteProblem([100], (0.0, 1.0), [1.0]), Direct(), death;
            rng = StableRNG(1))
        # run the failing solve once, then inspect the captured error
        err = try
            solve(jp, BoundedSSA(; rate_bound = 50.0));
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("BoundedSSA rate_bound violated", msg)   # identifying diagnostic
        @test occursin("50.0", msg)                             # the supplied rate bound Λ
        @test occursin("100.0", msg)                            # the offending total propensity
    end
end
