using JumpProcesses, Test, StableRNGs, ForwardDiff

regular_leaping_algs = (SimpleTauLeaping(),)

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
            adaptive in (false,)
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

@testset "Native mass-action operations" begin
    for scale_rates in (true, false)
        maj = MassActionJump(
            [2.0, 6.0], [Pair{Int, Int}[], [1 => 3]],
            [[1 => 1], [1 => -3, 2 => 1]]; scale_rates
        )
        jp = JumpProblem(DiscreteProblem([5.0, 0.0], (0.0, 1.0)), PureLeaping(), maj)
        @test jp.regular_jump === nothing
        out = fill(NaN, 2)
        massaction_rates!(out, jp.massaction_jump, [5.0, 0.0])
        @test out == [2.0, scale_rates ? 60.0 : 360.0]
        massaction_rates!(out, jp.massaction_jump, [1.0, 0.0])
        @test out == [2.0, 0.0]
        du = fill(NaN, 2)
        massaction_stoichiometry_mul!(du, jp.massaction_jump, [4.0, 2.0])
        @test du == [-2.0, 2.0]
        u = [5.0, 0.0]
        massaction_rates!(out, jp.massaction_jump, u)
        massaction_stoichiometry_mul!(du, jp.massaction_jump, out)
        @test massaction_drift!(zeros(2), jp.massaction_jump, u) == du
        jac = ForwardDiff.jacobian(u) do x
            massaction_drift!(similar(x), jp.massaction_jump, x)
        end
        factor = scale_rates ? 1.0 : 6.0
        @test jac == [-141.0factor 0.0; 47.0factor 0.0]
    end
    maj = MassActionJump([[1 => 1]], [[1 => -1, 2 => 1]]; param_idxs = [1])
    jp = JumpProblem(DiscreteProblem([20.0, 0.0], (0.0, 1.0), [0.1]), PureLeaping(), maj)
    remade = remake(jp; p = [0.3])
    out = zeros(1)
    massaction_rates!(out, remade.massaction_jump, remade.prob.u0)
    @test out == [6.0]
    @test remade.regular_jump === nothing
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

@testset "Single-reaction mass-action operations" begin
    for (rate, reactants, net) in (
            (6.0, [1 => 3], [1 => -3, 2 => 1]),
            (2.0, Pair{Int, Int}[], [1 => 1]),
        )
        single = MassActionJump(rate, reactants, net)
        vector = MassActionJump([rate], [reactants], [net])
        u = [5.0, 0.0]
        @test massaction_rates!(zeros(1), single, u) == massaction_rates!(zeros(1), vector, u)
        @test massaction_stoichiometry_mul!(zeros(2), single, [2.0]) ==
            massaction_stoichiometry_mul!(zeros(2), vector, [2.0])
        @test massaction_drift!(zeros(2), single, u) == massaction_drift!(zeros(2), vector, u)
    end
end
