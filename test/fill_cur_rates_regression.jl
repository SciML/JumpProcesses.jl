using JumpProcesses
using Statistics, Random, Test

# Regression for the shared `fill_cur_rates!` refactor (src/aggregators/direct.jl):
# the raw rate-fill was factored out of `time_to_next_jump` so the StochasticAD
# bounded SSA path can reuse it. The stock Direct / DirectFW SSA path must be
# UNCHANGED by that. This covers mass-action rates (`evalrxrate`), multi-channel
# cumulative selection (`searchsortedfirst`), the mixed mass-action + constant
# ordering, and the function-wrapper (`DirectFW`) path. Uses only JumpProcesses
# (no OrdinaryDiffEq), so it runs in the isolated StochasticAD env alongside the
# change it guards.

# final value of species 1 over `nT` independent, seeded trajectories
function final_species1(jprob, alg; nT, seedbase = 0)
    xs = Vector{Float64}(undef, nT)
    for i in 1:nT
        xs[i] = solve(jprob, alg; seed = seedbase + i).u[end][1]
    end
    return xs
end

@testset "fill_cur_rates! stock-path regression (Direct/DirectFW)" begin
    T = 1.0

    # --- mass-action, single channel: X --(μ)--> 0, rate μ·X -----------------
    # E[X(T)] = X0·e^{-μT}
    @testset "mass-action pure death (Direct)" begin
        X0, μ = 100, 0.5
        maj = MassActionJump([μ], [[1 => 1]], [[1 => -1]])
        jp = JumpProblem(DiscreteProblem([X0], (0.0, T)), Direct(), maj)
        analytic = X0 * exp(-μ * T)
        xs = final_species1(jp, SSAStepper(); nT = 5000)
        @test abs(mean(xs) - analytic) < 4 * std(xs) / sqrt(length(xs))
    end

    # --- mass-action, multi-channel: A <-(k1,k2)-> B (both first order) -------
    # linear system => the mean is exact: A_ss = k2·N/(k1+k2),
    # E[A(T)] = A_ss + (A0 - A_ss)·e^{-(k1+k2)T}. Exercises evalrxrate over two
    # channels and the cumulative-rate channel selection.
    @testset "mass-action reversible two-channel (Direct)" begin
        A0, B0, k1, k2 = 80, 20, 0.8, 0.4
        N = A0 + B0
        maj = MassActionJump([k1, k2],
            [[1 => 1], [2 => 1]],
            [[1 => -1, 2 => 1], [1 => 1, 2 => -1]])
        jp = JumpProblem(DiscreteProblem([A0, B0], (0.0, T)), Direct(), maj)
        Ass = k2 * N / (k1 + k2)
        analytic = Ass + (A0 - Ass) * exp(-(k1 + k2) * T)
        xs = final_species1(jp, SSAStepper(); nT = 5000)
        @test abs(mean(xs) - analytic) < 4 * std(xs) / sqrt(length(xs))
        @test all(x -> 0 <= x <= N, xs)   # conservation A+B=N keeps A in [0,N]
    end

    # --- mixed mass-action + constant jump: birth (constant λ) + death (μ·X) --
    # fills mass-action rate first, then the constant-jump rate, in fill_cur_rates!.
    # E[X(T)] = λ/μ + (X0 - λ/μ)·e^{-μT}
    @testset "mixed mass-action death + constant birth (Direct)" begin
        X0, λ, μ = 50, 10.0, 0.3
        birth = ConstantRateJump((u, p, t) -> λ, integ -> (integ.u[1] += 1; nothing))
        death = MassActionJump([μ], [[1 => 1]], [[1 => -1]])
        jp = JumpProblem(DiscreteProblem([X0], (0.0, T)), Direct(), birth, death)
        analytic = λ / μ + (X0 - λ / μ) * exp(-μ * T)
        xs = final_species1(jp, SSAStepper(); nT = 5000)
        @test abs(mean(xs) - analytic) < 4 * std(xs) / sqrt(length(xs))
    end

    # --- same mixed model via the function-wrapper aggregator -----------------
    # exercises the AbstractArray fill_cur_rates! / time_to_next_jump methods.
    @testset "mixed model via DirectFW (fwrapper path)" begin
        X0, λ, μ = 50, 10.0, 0.3
        birth = ConstantRateJump((u, p, t) -> λ, integ -> (integ.u[1] += 1; nothing))
        death = MassActionJump([μ], [[1 => 1]], [[1 => -1]])
        jp = JumpProblem(DiscreteProblem([X0], (0.0, T)), DirectFW(), birth, death)
        analytic = λ / μ + (X0 - λ / μ) * exp(-μ * T)
        xs = final_species1(jp, SSAStepper(); nT = 5000)
        @test abs(mean(xs) - analytic) < 4 * std(xs) / sqrt(length(xs))
    end

    # --- Direct and DirectFW must produce IDENTICAL trajectories --------------
    # same rates, same cumulative sums, same RNG draws => bit-identical paths.
    # Deterministic check (no MC tolerance): the sharpest guard that both
    # fill_cur_rates! methods compute the same per-channel and cumulative rates.
    @testset "Direct == DirectFW (identical trajectories)" begin
        X0, λ, μ = 40, 8.0, 0.5
        mkbirth() = ConstantRateJump((u, p, t) -> λ, integ -> (integ.u[1] += 1; nothing))
        mkdeath() = MassActionJump([μ], [[1 => 1]], [[1 => -1]])
        jp_d = JumpProblem(DiscreteProblem([X0], (0.0, T)), Direct(), mkbirth(), mkdeath())
        jp_f = JumpProblem(DiscreteProblem([X0], (0.0, T)), DirectFW(), mkbirth(), mkdeath())
        for i in 1:25
            sd = solve(jp_d, SSAStepper(); seed = i)
            sf = solve(jp_f, SSAStepper(); seed = i)
            @test sd.u[end] == sf.u[end]
        end
    end
end
