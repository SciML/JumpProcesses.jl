using JumpProcesses, Test, ForwardDiff

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
            actual = JumpProblem(prob, PureLeaping(), maj)
            reference = JumpProblem(prob, PureLeaping(), rj)
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
    massaction_rates!(out, remade.massaction_jump, remade.prob.u0; p = remade.prob.p)
    @test out == [6.0]
    @test massaction_drift!(
        zeros(2), remade.massaction_jump, remade.prob.u0;
        p = remade.prob.p
    ) == [-6.0, 6.0]
    @test remade.massaction_jump.scaled_rates === nothing
    @test successful_retcode(solve(remade, SimpleTauLeaping(); dt = 0.01, seed = 123))
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

@testset "Large integer mass-action populations" begin
    n = 3_000_000
    for single in (false, true), (rate, scale_rates) in ((1.0, true), (6, true), (1, false))
        reference = Float64(big(n) * (n - 1) * (n - 2)) * (scale_rates ? rate / 6 : rate)
        maj = single ? MassActionJump(rate, [1 => 3], [1 => -3]; scale_rates) :
            MassActionJump([rate], [[1 => 3]], [[1 => -3]]; scale_rates)
        @test only(massaction_rates!(zeros(1), maj, [n])) ≈ reference
        @test only(massaction_drift!(zeros(1), maj, [n])) ≈ -3 * reference
    end
end

@testset "Integer reaction counts with floating-point output" begin
    maj = MassActionJump([1], [[1 => 1]], [[1 => -3]]; scale_rates = false)
    count = typemax(Int)
    @test massaction_stoichiometry_mul!(zeros(1), maj, [count]) == [-3 * Float64(count)]
end

@testset "Single-reaction parameter-mapped operations" begin
    for (reactants, net, u, expected_rate, expected_drift) in (
                ([1 => 1], [1 => -1, 2 => 1], [20, 0], 6.0, [-6.0, 6.0]),
                ([1 => 3], [1 => -3, 2 => 1], [5, 0], 3.0, [-9.0, 3.0]),
                (Pair{Int, Int}[], [1 => 1], [0], 0.3, [0.3]),
            ), scale_rates in (true, false)
        maj = MassActionJump(reactants, net; param_idxs = 1, scale_rates)
        vector = MassActionJump([reactants], [net]; param_idxs = [1], scale_rates)
        rate = scale_rates || length(reactants) == 0 ? expected_rate :
            expected_rate * factorial(only(reactants).second)
        drift = scale_rates || length(reactants) == 0 ? expected_drift :
            expected_drift * factorial(only(reactants).second)
        @test massaction_rates!(zeros(1), maj, u; p = [0.3]) ≈ [rate]
        @test massaction_drift!(zeros(length(u)), maj, u; p = [0.3]) ≈ drift
        @test massaction_stoichiometry_mul!(zeros(length(u)), maj, [2.0]) ==
            massaction_stoichiometry_mul!(zeros(length(u)), vector, [2.0])
        @test massaction_rates!(zeros(1), maj, u; p = [0.6]) ≈ [2rate]
        @test maj.scaled_rates === nothing
    end
end

@testset "Integer rate constants in fixed-step leaping" begin
    for scale_rates in (true, false)
        rate = scale_rates ? 6 : 1
        integer_jump = MassActionJump([rate], [[1 => 3]], [[1 => -3]]; scale_rates)
        float_jump = MassActionJump([Float64(rate)], [[1 => 3]], [[1 => -3]]; scale_rates)
        prob = DiscreteProblem([3_000_000], (0.0, 1.0e-18))
        actual = solve(
            JumpProblem(prob, PureLeaping(), integer_jump), SimpleTauLeaping();
            dt = 1.0e-18, seed = 123
        )
        expected = solve(
            JumpProblem(prob, PureLeaping(), float_jump), SimpleTauLeaping();
            dt = 1.0e-18, seed = 123
        )
        @test successful_retcode(actual)
        @test actual.u == expected.u
    end
end
