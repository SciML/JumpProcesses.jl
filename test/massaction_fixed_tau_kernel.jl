using JumpProcesses, KernelAbstractions, Adapt, Test, Statistics

function test_massaction_fixed_tau_kernel(backend)
    return @testset "Fixed-step mass-action kernel ($T)" for T in (Float64, Int)
        maj = MassActionJump([0.2], [[1 => 1]], [[1 => -1, 2 => 1]])
        jp = JumpProblem(DiscreteProblem(T[1001, 0], (0.0, 1.0)), PureLeaping(), maj)
        sol = solve(
            EnsembleProblem(jp), SimpleTauLeaping(), EnsembleGPUKernel(backend);
            trajectories = 2000, dt = 0.01
        )
        @test length(sol.u) == 2000
        @test all(s -> successful_retcode(s), sol.u)
        @test all(s -> s.t[end] ≈ 1.0, sol.u)
        @test all(s -> all(u -> sum(u) ≈ 1001, s.u), sol.u)
        @test mean(s.u[end][1] for s in sol.u) ≈ 1001 * (1 - 0.2 * 0.01)^100 rtol = 0.01

        births = MassActionJump([10.0], [Pair{Int, Int}[]], [[1 => 1]])
        jp = JumpProblem(DiscreteProblem(T[0], (0.0, 1.0)), PureLeaping(), births)
        sol = solve(
            EnsembleProblem(jp), SimpleTauLeaping(), EnsembleGPUKernel(backend);
            trajectories = 2000, dt = 0.01
        )
        @test mean(s.u[end][1] for s in sol.u) ≈ 10 rtol = 0.05
    end
end
