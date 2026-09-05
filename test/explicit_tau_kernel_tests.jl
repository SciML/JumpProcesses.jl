using JumpProcesses
using Test, Statistics
using KernelAbstractions, Adapt

# Body of the SimpleExplicitTauLeaping kernel tests, run against both a GPU
# backend and the KernelAbstractions CPU backend so the same behaviour is covered
# without a GPU.
#
# The reference is the serial SimpleExplicitTauLeaping, not the exact SSA: tau
# leaping carries a discretization bias that grows with `epsilon`, and the kernel
# is expected to reproduce the host solver including that bias. One test below
# pins the bias down by checking it shrinks as `epsilon` shrinks.
function run_explicit_tau_kernel_tests(backend, nsims)
    leaping_prob(rates, rs, ns, u0, tspan) = JumpProblem(DiscreteProblem(u0, tspan),
        PureLeaping(), MassActionJump(rates, rs, ns))

    kernel_solve(jp, nsims; kwargs...) = solve(EnsembleProblem(jp),
        SimpleExplicitTauLeaping(), EnsembleGPUKernel(backend);
        trajectories = nsims, kwargs...)
    serial_solve(jp, nsims; kwargs...) = solve(EnsembleProblem(jp),
        SimpleExplicitTauLeaping(), EnsembleSerial(); trajectories = nsims, kwargs...)

    # Compare the mean of every species at every saved time.
    function compare_means(jp, nsims, nspec; rtol = 0.05, atol = 0.5, kwargs...)
        sk = kernel_solve(jp, nsims; kwargs...)
        ss = serial_solve(jp, nsims; kwargs...)
        @test sk.u[1].t == ss.u[1].t
        for k in eachindex(sk.u[1].t), s in 1:nspec
            mk = mean(sk.u[i].u[k][s] for i in 1:nsims)
            ms = mean(ss.u[i].u[k][s] for i in 1:nsims)
            @test isapprox(mk, ms; rtol, atol)
        end
        return sk
    end

    # Linear death X -> 0.
    let
        jp = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [1000.0], (0.0, 4.0))
        sol = compare_means(jp, nsims, 1; saveat = 1.0)
        @test sol.u[1].t == [0.0, 1.0, 2.0, 3.0, 4.0]
        @test sol.u[1].u[1] == [1000.0]
        # the population can only fall
        @test all(issorted(reverse([sol.u[i].u[k][1] for k in 1:5])) for i in 1:100)
    end

    # Zero order 0 -> A, i.e. an empty reactant stoichiometry, so max_hor is 0
    # for the product species and g_i takes its default value.
    let
        jp = leaping_prob([5.0], [Pair{Int, Int}[]], [[1 => 1]], [0.0], (0.0, 4.0))
        sol = compare_means(jp, nsims, 1; saveat = 2.0)
        # A(t) is Poisson with mean 5t here, and the leap is exact for a constant rate
        for (k, t) in enumerate(sol.u[1].t)
            @test isapprox(mean(sol.u[i].u[k][1] for i in 1:nsims), 5.0 * t,
                rtol = 0.05, atol = 1.0e-8)
        end
    end

    # SIR.
    let
        jp = leaping_prob([0.1 / 1000, 0.01], [[1 => 1, 2 => 1], [2 => 1]],
            [[1 => -1, 2 => 1], [2 => -1, 3 => 1]], [999.0, 10.0, 0.0], (0.0, 250.0))
        compare_means(jp, nsims, 3; saveat = 50.0)
    end

    # Second order 2A -> B. Exercises the falling factorial in the propensity and
    # the g_i branch for max_hor 2 with max_stoich 2.
    let
        jp = leaping_prob([0.01], [[1 => 2]], [[1 => -2, 2 => 1]], [1000.0, 0.0],
            (0.0, 5.0))
        sol = compare_means(jp, nsims, 2; saveat = 1.0)
        # A is consumed two at a time, so its parity is conserved along a path
        @test all(all(iseven, (Int(sol.u[i].u[k][1]) for k in 1:6)) for i in 1:100)
    end

    # Third order 3A -> B, for the deeper g_i branch (max_hor 3, max_stoich 3).
    let
        jp = leaping_prob([1.0e-6], [[1 => 3]], [[1 => -3, 2 => 1]], [1000.0, 0.0],
            (0.0, 5.0))
        compare_means(jp, nsims, 2; saveat = 2.5)
    end

    # Reversible binding A + B <-> C, a multi species model with conservation laws.
    let
        jp = leaping_prob([0.001, 0.5], [[1 => 1, 2 => 1], [3 => 1]],
            [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]],
            [1000.0, 800.0, 0.0], (0.0, 10.0))
        sol = compare_means(jp, nsims, 3; saveat = 5.0)
        # A + C and B + C are conserved by both reactions
        @test all(sol.u[i].u[k][1] + sol.u[i].u[k][3] == 1000.0 for i in 1:100, k in 1:3)
        @test all(sol.u[i].u[k][2] + sol.u[i].u[k][3] == 800.0 for i in 1:100, k in 1:3)
    end

    # Extinction. Once every molecule is gone no reaction can fire, and the path
    # has to stay constant for the rest of the grid.
    let
        jp = leaping_prob([2.0], [[1 => 1]], [[1 => -1]], [3.0], (0.0, 20.0))
        sol = kernel_solve(jp, 1000; saveat = 5.0)
        @test all(sol.u[i].u[end][1] == 0.0 for i in 1:1000)
        @test all(issorted(reverse([sol.u[i].u[k][1] for k in eachindex(sol.u[i].t)]))
        for i in 1:1000)
    end

    # The leap size has to respond to epsilon: a tighter tolerance must move the
    # mean towards the exact SSA result.
    let
        c, x0, tend = 0.5, 1000.0, 4.0
        jp = leaping_prob([c], [[1 => 1]], [[1 => -1]], [x0], (0.0, tend))
        exact = x0 * exp(-c * tend)
        errs = map((0.05, 0.01, 0.002)) do epsilon
            sol = solve(EnsembleProblem(jp), SimpleExplicitTauLeaping(; epsilon),
                EnsembleGPUKernel(backend); trajectories = nsims, saveat = tend)
            abs(mean(sol.u[i].u[end][1] for i in 1:nsims) - exact)
        end
        @test errs[1] > errs[2] > errs[3]
        @test errs[3] < 0.02 * exact
    end

    # saveat handling and the save_start / save_end flags.
    let
        jp = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [500.0], (0.0, 3.0))

        sol = kernel_solve(jp, 10; saveat = 1.0)
        @test sol.u[1].t == [0.0, 1.0, 2.0, 3.0]
        @test sol.u[1].u[1] == [500.0]

        sol = kernel_solve(jp, 10; saveat = [0.5, 1.5, 2.5])
        @test sol.u[1].t == [0.0, 0.5, 1.5, 2.5, 3.0]

        sol = kernel_solve(jp, 10; saveat = [0.5, 1.5], save_start = false,
            save_end = false)
        @test sol.u[1].t == [0.5, 1.5]
    end

    # Unsupported inputs must be rejected rather than silently producing wrong results.
    let
        # a RegularJump cannot be evaluated on the device
        rj = RegularJump((out, u, p, t) -> (out[1] = 0.5u[1]),
            (du, u, p, t, counts, mark) -> (du[1] = -counts[1]), 1)
        rj_prob = JumpProblem(DiscreteProblem([50.0], (0.0, 3.0)), PureLeaping(), rj)
        @test_throws ErrorException solve(EnsembleProblem(rj_prob),
            SimpleExplicitTauLeaping(), EnsembleGPUKernel(backend);
            trajectories = 10, saveat = 1.0)

        maj_prob = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [50.0], (0.0, 3.0))

        # saveat is mandatory: the leap size adapts, so the step count is unknown
        @test_throws ErrorException solve(EnsembleProblem(maj_prob),
            SimpleExplicitTauLeaping(), EnsembleGPUKernel(backend); trajectories = 10)

        @test_throws ErrorException solve(EnsembleProblem(maj_prob),
            SimpleExplicitTauLeaping(), EnsembleGPUKernel(backend); trajectories = 10,
            saveat = 1.0,
            callback = DiscreteCallback((u, t, integrator) -> false,
                integrator -> nothing))

        # the reaction data is shared by all threads, so trajectories cannot differ
        varying = EnsembleProblem(maj_prob; prob_func = (prob, ctx) -> prob)
        @test_throws ErrorException solve(varying, SimpleExplicitTauLeaping(),
            EnsembleGPUKernel(backend); trajectories = 5, saveat = 1.0)
    end

    return nothing
end
