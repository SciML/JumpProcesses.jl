using JumpProcesses
using Test, Statistics
using KernelAbstractions, Adapt
using StaticArrays

# Body of the adaptive tau-leaping kernel tests, run against both a GPU backend
# and the KernelAbstractions CPU backend.
#
# As with the other leaping kernels the reference is the serial solver of the
# same algorithm, and it runs fewer trajectories than the kernel because a stiff
# model drives it into the nonlinear solve on most steps.
function run_adaptive_tau_kernel_tests(backend, nsims)
    leaping_prob(rates, rs, ns, u0, tspan) = JumpProblem(DiscreteProblem(u0, tspan),
        PureLeaping(), MassActionJump(rates, rs, ns))

    implicit_algs = (SimpleImplicitTauLeaping(), SimpleTrapezoidalLeaping())

    function compare_means(jp, alg, nsims, nspec; nref = min(nsims, 2000),
            rtol = 0.05, atol = 0.5, kwargs...)
        sk = solve(EnsembleProblem(jp), alg, EnsembleGPUKernel(backend);
            trajectories = nsims, kwargs...)
        ss = solve(EnsembleProblem(jp), alg, EnsembleSerial();
            trajectories = nref, kwargs...)
        @test sk.u[1].t == ss.u[1].t
        for k in eachindex(sk.u[1].t), s in 1:nspec
            mk = mean(sk.u[i].u[k][s] for i in 1:nsims)
            ms = mean(ss.u[i].u[k][s] for i in 1:nref)
            @test isapprox(mk, ms; rtol, atol)
        end
        return sk
    end

    # The stiffness test has to actually discriminate, or the implicit branch
    # below is never taken and the rest of this file passes without exercising it.
    let
        ext = Base.get_extension(JumpProcesses, :JumpProcessesKernelAbstractionsExt)
        is_stiff_at(rates, rs, ns, u0v) = begin
            maj = MassActionJump(collect(Float64.(rates)), rs, ns)
            majg = ext.GPUMassActionJump(maj, CPU(), Float64)
            ext.gpu_is_stiff(SVector{length(u0v), Float64}(Float64.(u0v)), majg,
                0.05, Float64)
        end

        # fast reversible pair with a slow drain: stiff everywhere along the path
        for u in ([1000.0, 1000.0, 0.0], [1000.0, 900.0, 100.0], [500.0, 500.0, 1000.0])
            @test is_stiff_at([100.0, 100.0, 0.1], [[1 => 1], [2 => 1], [1 => 1]],
                [[1 => -1, 2 => 1], [1 => 1, 2 => -1], [1 => -1, 3 => 1]], u)
        end

        # a single decay, and SIR, are not stiff
        for u in ([1000.0], [100.0], [10.0])
            @test !is_stiff_at([0.5], [[1 => 1]], [[1 => -1]], u)
        end
        @test !is_stiff_at([0.1 / 1000, 0.01], [[1 => 1, 2 => 1], [2 => 1]],
            [[1 => -1, 2 => 1], [2 => -1, 3 => 1]], [999.0, 10.0, 0.0])
    end

    # A stiff model, where the switch should be taking implicit steps, run with
    # both implicit formulations.
    for implicit_alg in implicit_algs
        jp = leaping_prob([100.0, 100.0, 0.1], [[1 => 1], [2 => 1], [1 => 1]],
            [[1 => -1, 2 => 1], [1 => 1, 2 => -1], [1 => -1, 3 => 1]],
            [1000.0, 1000.0, 0.0], (0.0, 1.0))
        sol = compare_means(jp, SimpleAdaptiveTauLeaping(; implicit_alg), nsims, 3;
            saveat = 0.25)
        @test all(sum(sol.u[i].u[k]) == 2000.0 for i in 1:100, k in 1:5)
    end

    # A non-stiff model, where it should stay on the explicit branch.
    let
        jp = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [1000.0], (0.0, 4.0))
        sol = compare_means(jp, SimpleAdaptiveTauLeaping(), nsims, 1; saveat = 1.0)
        @test sol.u[1].t == [0.0, 1.0, 2.0, 3.0, 4.0]
        @test all(issorted(reverse([sol.u[i].u[k][1] for k in 1:5])) for i in 1:100)
    end

    # SIR and SEIR, the latter needing a four-species implicit solve if the
    # switch ever takes that branch.
    let
        jp = leaping_prob([0.1 / 1000, 0.01], [[1 => 1, 2 => 1], [2 => 1]],
            [[1 => -1, 2 => 1], [2 => -1, 3 => 1]], [999.0, 10.0, 0.0], (0.0, 100.0))
        compare_means(jp, SimpleAdaptiveTauLeaping(), nsims, 3; saveat = 50.0)
    end
    let
        jp = leaping_prob([0.3 / 1000, 0.2, 0.1],
            [[1 => 1, 3 => 1], [2 => 1], [3 => 1]],
            [[1 => -1, 2 => 1], [2 => -1, 3 => 1], [3 => -1, 4 => 1]],
            [999.0, 0.0, 10.0, 0.0], (0.0, 20.0))
        sol = compare_means(jp, SimpleAdaptiveTauLeaping(), nsims, 4; saveat = 10.0)
        @test all(sum(sol.u[i].u[k]) == 1009.0 for i in 1:100, k in 1:3)
    end

    # Second order, for the falling factorial and its derivative.
    let
        jp = leaping_prob([0.01], [[1 => 2]], [[1 => -2, 2 => 1]], [200.0, 0.0],
            (0.0, 2.0))
        sol = compare_means(jp, SimpleAdaptiveTauLeaping(), nsims, 2; saveat = 1.0)
        @test all(all(iseven, (Int(sol.u[i].u[k][1]) for k in 1:3)) for i in 1:100)
    end

    # Conservation laws.
    let
        jp = leaping_prob([0.001, 0.5], [[1 => 1, 2 => 1], [3 => 1]],
            [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]],
            [1000.0, 800.0, 0.0], (0.0, 10.0))
        sol = compare_means(jp, SimpleAdaptiveTauLeaping(), nsims, 3; saveat = 5.0)
        @test all(sol.u[i].u[k][1] + sol.u[i].u[k][3] == 1000.0 for i in 1:100, k in 1:3)
        @test all(sol.u[i].u[k][2] + sol.u[i].u[k][3] == 800.0 for i in 1:100, k in 1:3)
    end

    # Extinction.
    let
        jp = leaping_prob([2.0], [[1 => 1]], [[1 => -1]], [3.0], (0.0, 20.0))
        sol = solve(EnsembleProblem(jp), SimpleAdaptiveTauLeaping(),
            EnsembleGPUKernel(backend); trajectories = 1000, saveat = 5.0)
        @test all(sol.u[i].u[end][1] == 0.0 for i in 1:1000)
    end

    # saveat handling.
    let
        jp = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [500.0], (0.0, 3.0))
        kern(; kwargs...) = solve(EnsembleProblem(jp), SimpleAdaptiveTauLeaping(),
            EnsembleGPUKernel(backend); trajectories = 10, kwargs...)

        sol = kern(; saveat = 1.0)
        @test sol.u[1].t == [0.0, 1.0, 2.0, 3.0]
        @test sol.u[1].u[1] == [500.0]

        sol = kern(; saveat = [0.5, 1.5, 2.5])
        @test sol.u[1].t == [0.0, 0.5, 1.5, 2.5, 3.0]

        sol = kern(; saveat = [0.5, 1.5], save_start = false, save_end = false)
        @test sol.u[1].t == [0.5, 1.5]
    end

    # Unsupported inputs must be rejected rather than silently producing wrong results.
    let
        maj_prob = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [50.0], (0.0, 3.0))

        # the eigenvalue stiffness test has no kernel-safe form
        @test_throws ErrorException solve(EnsembleProblem(maj_prob),
            SimpleAdaptiveTauLeaping(eigenvalue_check = true),
            EnsembleGPUKernel(backend); trajectories = 10, saveat = 1.0)

        rj = RegularJump((out, u, p, t) -> (out[1] = 0.5u[1]),
            (du, u, p, t, counts, mark) -> (du[1] = -counts[1]), 1)
        rj_prob = JumpProblem(DiscreteProblem([50.0], (0.0, 3.0)), PureLeaping(), rj)
        @test_throws ErrorException solve(EnsembleProblem(rj_prob),
            SimpleAdaptiveTauLeaping(), EnsembleGPUKernel(backend);
            trajectories = 10, saveat = 1.0)

        @test_throws ErrorException solve(EnsembleProblem(maj_prob),
            SimpleAdaptiveTauLeaping(), EnsembleGPUKernel(backend); trajectories = 10)

        @test_throws ErrorException solve(EnsembleProblem(maj_prob),
            SimpleAdaptiveTauLeaping(), EnsembleGPUKernel(backend); trajectories = 10,
            saveat = 1.0,
            callback = DiscreteCallback((u, t, integrator) -> false,
                integrator -> nothing))

        varying = EnsembleProblem(maj_prob; prob_func = (prob, ctx) -> prob)
        @test_throws ErrorException solve(varying, SimpleAdaptiveTauLeaping(),
            EnsembleGPUKernel(backend); trajectories = 5, saveat = 1.0)
    end

    return nothing
end
