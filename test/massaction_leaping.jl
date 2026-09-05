using JumpProcesses, StochasticDiffEq, Test, StableRNGs

regular_leaping_algs = (
    SimpleTauLeaping(), TauLeaping(), CaoTauLeaping(),
    ImplicitTauLeaping(), ThetaTrapezoidalTauLeaping(),
)

@testset "Mass-action input for regular leaping solvers" begin
    maj = MassActionJump(
        [0.02, 0.1], [[1 => 2], [2 => 1]],
        [[1 => -2, 2 => 1], [1 => 2, 2 => -1]]
    )
    prob = DiscreteProblem([40.0, 20.0], (0.0, 0.2))
    function rate!(out, u, p, t)
        out[1] = u[1] * max(u[1] - 1, 0) * 0.01
        out[2] = 0.1 * u[2]
        nothing
    end
    function change!(du, u, p, t, counts, mark)
        du[1] = -2 * counts[1] + 2 * counts[2]
        du[2] = counts[1] - counts[2]
        nothing
    end
    rj = RegularJump(rate!, change!, 2)
    for alg in regular_leaping_algs,
            adaptive in (alg isa Union{TauLeaping, CaoTauLeaping} ? (false, true) : (false,))
        @testset "$(nameof(typeof(alg))), adaptive=$adaptive" begin
            actual = JumpProblem(prob, PureLeaping(), maj; rng = StableRNG(123))
            reference = JumpProblem(prob, PureLeaping(), rj; rng = StableRNG(123))
            opts = alg isa SimpleTauLeaping ? (; dt = 0.005, seed = 123) :
                (; dt = 0.005, seed = 123, adaptive)
            sol = solve(actual, alg; opts...)
            ref = solve(reference, alg; opts...)
            @test successful_retcode(sol)
            @test sol.t == ref.t
            @test sol.u == ref.u
            @test all(u -> u[1] + 2u[2] ≈ 80, sol.u)
        end
    end
end

@testset "Generated mass-action rate and update" begin
    for scale_rates in (true, false)
        maj = MassActionJump(
            [2.0, 6.0], [Pair{Int, Int}[], [1 => 3]],
            [[1 => 1], [1 => -3, 2 => 1]]; scale_rates
        )
        jp = JumpProblem(DiscreteProblem([5.0, 0.0], (0.0, 1.0)), PureLeaping(), maj)
        out = fill(NaN, 2)
        jp.regular_jump.rate(out, [5.0, 0.0], nothing, 0.0)
        @test out == [2.0, scale_rates ? 60.0 : 360.0]
        jp.regular_jump.rate(out, [1.0, 0.0], nothing, 0.0)
        @test out == [2.0, 0.0]
        du = fill(NaN, 2)
        jp.regular_jump.c(du, [5.0, 0.0], nothing, 0.0, [4.0, 2.0], nothing)
        @test du == [-2.0, 2.0]
    end
    maj = MassActionJump([[1 => 1]], [[1 => -1, 2 => 1]]; param_idxs = [1])
    jp = JumpProblem(DiscreteProblem([20.0, 0.0], (0.0, 1.0), [0.1]), PureLeaping(), maj)
    remade = remake(jp; p = [0.3])
    out = zeros(1)
    remade.regular_jump.rate(out, remade.prob.u0, remade.prob.p, 0.0)
    @test out == [6.0]
    oprob = ODEProblem((du, u, p, t) -> fill!(du, 0), [20.0, 0.0], (0.0, 1.0), [0.1])
    ojp = JumpProblem(oprob, PureLeaping(), maj)
    @test ojp.regular_jump === nothing
    @test_throws ErrorException solve(ojp, SimpleTauLeaping(); dt = 0.01)
end

@testset "General regular rates remain supported" begin
    rate!(out, u, p, t) = (out[1] = 100 * u[1] / (1 + u[1]))
    function change!(du, u, p, t, counts, mark)
        du[1] = -counts[1]
        du[2] = counts[1]
        nothing
    end
    jp = JumpProblem(
        DiscreteProblem([100.0, 0.0], (0.0, 0.1)),
        PureLeaping(), RegularJump(rate!, change!, 1)
    )
    for alg in regular_leaping_algs
        opts = alg isa SimpleTauLeaping ? (; dt = 0.001, seed = 42) :
            (; dt = 0.001, seed = 42, adaptive = false)
        sol = solve(jp, alg; opts...)
        @test successful_retcode(sol)
        @test sol.u[end][2] > 0
        @test all(u -> sum(u) ≈ 100, sol.u)
    end
end
