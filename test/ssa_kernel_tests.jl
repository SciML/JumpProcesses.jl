using JumpProcesses
using Test, Statistics
using KernelAbstractions, Adapt

# Body of the SSA kernel tests, run against both a GPU backend and the
# KernelAbstractions CPU backend so the same behaviour is covered without a GPU.
#
# The GPU kernel samples each trajectory on the `saveat` grid and nothing else, so
# reference problems are built with `save_positions = (false, false)` to stop the
# serial solver from also saving at every jump. That makes the two `t` vectors
# line up index for index.
function run_ssa_kernel_tests(backend, nsims)
    majump_prob(rates, rs, ns, u0, tspan) = JumpProblem(DiscreteProblem(u0, tspan),
        Direct(), MassActionJump(rates, rs, ns); save_positions = (false, false))

    # Linear death process X -> 0. E[X(t)] = X₀ exp(-ct) exactly, so this pins the
    # waiting time distribution down against an analytic result rather than another
    # sampler.
    let
        c = 0.5
        X₀ = 100
        jump_prob = majump_prob([c], [[1 => 1]], [[1 => -1]], [X₀], (0.0, 4.0))

        sol = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleGPUKernel(backend);
            trajectories = nsims, saveat = 1.0)

        @test sol.u[1].t == [0.0, 1.0, 2.0, 3.0, 4.0]
        for (k, t) in enumerate(sol.u[1].t)
            mean_kernel = mean(sol.u[i].u[k][1] for i in 1:nsims)
            @test isapprox(mean_kernel, X₀ * exp(-c * t), rtol = 0.02)
        end
    end

    # Zero order reaction 0 -> A, i.e. an empty reactant stoichiometry. A(t) is
    # Poisson with mean λt.
    let
        λ = 5.0
        jump_prob = majump_prob([λ], [Pair{Int, Int}[]], [[1 => 1]], [0], (0.0, 4.0))

        sol = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleGPUKernel(backend);
            trajectories = nsims, saveat = 2.0)

        for (k, t) in enumerate(sol.u[1].t)
            mean_kernel = mean(sol.u[i].u[k][1] for i in 1:nsims)
            @test isapprox(mean_kernel, λ * t, rtol = 0.02, atol = 1e-8)
        end
    end

    # SIR model, checked against the serial SSA.
    let
        β = 0.1 / 1000.0
        ν = 0.01
        jump_prob = majump_prob([β, ν], [[1 => 1, 2 => 1], [2 => 1]],
            [[1 => -1, 2 => 1], [2 => -1, 3 => 1]], [999, 10, 0], (0.0, 250.0))

        sol_kernel = solve(EnsembleProblem(jump_prob), SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = nsims, saveat = 50.0)
        sol_serial = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial();
            trajectories = nsims, saveat = 50.0)

        @test sol_kernel.u[1].t == sol_serial.u[1].t
        for k in eachindex(sol_kernel.u[1].t), s in 1:3
            mean_kernel = mean(sol_kernel.u[i].u[k][s] for i in 1:nsims)
            mean_serial = mean(sol_serial.u[i].u[k][s] for i in 1:nsims)
            @test isapprox(mean_kernel, mean_serial, rtol = 0.05, atol = 0.5)
        end
    end

    # Second order reaction 2A -> B. Exercises the falling factorial A(A-1) in the
    # propensity and the combinatoric prefactor folded into `scaled_rates`.
    let
        jump_prob = majump_prob([0.01], [[1 => 2]], [[1 => -2, 2 => 1]], [100, 0], (0.0, 5.0))

        sol_kernel = solve(EnsembleProblem(jump_prob), SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = nsims, saveat = 1.0)
        sol_serial = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial();
            trajectories = nsims, saveat = 1.0)

        for k in eachindex(sol_kernel.u[1].t), s in 1:2
            mean_kernel = mean(sol_kernel.u[i].u[k][s] for i in 1:nsims)
            mean_serial = mean(sol_serial.u[i].u[k][s] for i in 1:nsims)
            @test isapprox(mean_kernel, mean_serial, rtol = 0.05, atol = 0.5)
        end

        # A is consumed two at a time, so parity of A is conserved along a path
        @test all(all(iseven, (sol_kernel.u[i].u[k][1] for k in 1:6)) for i in 1:100)
    end

    # Third order reaction 3A -> B, checking the deeper falling factorial.
    let
        jump_prob = majump_prob([1e-4], [[1 => 3]], [[1 => -3, 2 => 1]], [60, 0], (0.0, 5.0))

        sol_kernel = solve(EnsembleProblem(jump_prob), SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = nsims, saveat = 2.5)
        sol_serial = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial();
            trajectories = nsims, saveat = 2.5)

        for k in eachindex(sol_kernel.u[1].t)
            mean_kernel = mean(sol_kernel.u[i].u[k][1] for i in 1:nsims)
            mean_serial = mean(sol_serial.u[i].u[k][1] for i in 1:nsims)
            @test isapprox(mean_kernel, mean_serial, rtol = 0.05)
        end
    end

    # Reversible binding A + B <-> C, a multi species model with a conservation law.
    let
        jump_prob = majump_prob([0.01, 0.5], [[1 => 1, 2 => 1], [3 => 1]],
            [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]],
            [100, 80, 0], (0.0, 10.0))

        sol_kernel = solve(EnsembleProblem(jump_prob), SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = nsims, saveat = 5.0)
        sol_serial = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial();
            trajectories = nsims, saveat = 5.0)

        for k in eachindex(sol_kernel.u[1].t), s in 1:3
            mean_kernel = mean(sol_kernel.u[i].u[k][s] for i in 1:nsims)
            mean_serial = mean(sol_serial.u[i].u[k][s] for i in 1:nsims)
            @test isapprox(mean_kernel, mean_serial, rtol = 0.05, atol = 0.5)
        end

        # A + C and B + C are conserved by both reactions
        @test all(sol_kernel.u[i].u[k][1] + sol_kernel.u[i].u[k][3] == 100
        for i in 1:100, k in 1:3)
        @test all(sol_kernel.u[i].u[k][2] + sol_kernel.u[i].u[k][3] == 80
        for i in 1:100, k in 1:3)
    end

    # Extinction. Once every molecule is gone the total propensity is zero and the
    # path has to stay constant for the rest of the grid.
    let
        jump_prob = majump_prob([2.0], [[1 => 1]], [[1 => -1]], [3], (0.0, 20.0))

        sol = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleGPUKernel(backend);
            trajectories = 1000, saveat = 5.0)

        @test all(sol.u[i].u[end][1] == 0 for i in 1:1000)
        @test all(issorted(reverse([sol.u[i].u[k][1] for k in eachindex(sol.u[i].t)]))
        for i in 1:1000)
    end

    # saveat handling and the save_start / save_end flags.
    let
        jump_prob = majump_prob([0.5], [[1 => 1]], [[1 => -1]], [50], (0.0, 3.0))
        ensemble_prob = EnsembleProblem(jump_prob)

        sol = solve(ensemble_prob, SSAStepper(), EnsembleGPUKernel(backend);
            trajectories = 10, saveat = 1.0)
        @test sol.u[1].t == [0.0, 1.0, 2.0, 3.0]
        @test sol.u[1].u[1] == [50]

        sol = solve(ensemble_prob, SSAStepper(), EnsembleGPUKernel(backend);
            trajectories = 10, saveat = [0.5, 1.5, 2.5])
        @test sol.u[1].t == [0.0, 0.5, 1.5, 2.5, 3.0]

        sol = solve(ensemble_prob, SSAStepper(), EnsembleGPUKernel(backend);
            trajectories = 10, saveat = [0.5, 1.5], save_start = false, save_end = false)
        @test sol.u[1].t == [0.5, 1.5]
    end

    # Every trajectory must get its own random stream. If they shared one the sample
    # variance would collapse; compare against the exact Binomial(n, exp(-ct)) result.
    let
        n, c = 1000, 0.5
        jump_prob = majump_prob([c], [[1 => 1]], [[1 => -1]], [n], (0.0, 1.0))

        sol = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleGPUKernel(backend);
            trajectories = nsims, saveat = 1.0)
        finals = [sol.u[i].u[end][1] for i in 1:nsims]

        p = exp(-c)
        @test isapprox(mean(finals), n * p, rtol = 0.01)
        @test isapprox(var(finals), n * p * (1 - p), rtol = 0.05)
    end

    # Unsupported inputs must be rejected rather than silently producing wrong results.
    let
        rate(u, p, t) = 0.5u[1]
        affect!(integrator) = (integrator.u[1] -= 1)
        crj_prob = JumpProblem(DiscreteProblem([50], (0.0, 3.0)), Direct(),
            ConstantRateJump(rate, affect!))
        @test_throws ErrorException solve(EnsembleProblem(crj_prob), SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = 10, saveat = 1.0)

        maj_prob = majump_prob([0.5], [[1 => 1]], [[1 => -1]], [50], (0.0, 3.0))

        # saveat is mandatory: an SSA trajectory has no fixed number of steps
        @test_throws ErrorException solve(EnsembleProblem(maj_prob), SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = 10)

        @test_throws ErrorException solve(EnsembleProblem(maj_prob), SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = 10, saveat = 1.0,
            callback = DiscreteCallback((u, t, integrator) -> false, integrator -> nothing))

        # the reaction data is shared by all threads, so trajectories cannot differ
        varying = EnsembleProblem(maj_prob; prob_func = (prob, ctx) -> prob)
        @test_throws ErrorException solve(varying, SSAStepper(),
            EnsembleGPUKernel(backend); trajectories = 5, saveat = 1.0)
    end
    return nothing
end
