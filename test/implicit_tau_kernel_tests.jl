using JumpProcesses
using Test, Statistics
using KernelAbstractions, Adapt

# Body of the implicit tau-leaping kernel tests, run against both a GPU backend
# and the KernelAbstractions CPU backend so the same behaviour is covered without
# a GPU. Both implicit formulations are exercised.
#
# The reference is the serial solver of the same algorithm, not the exact SSA:
# tau leaping carries a discretization bias the kernel is expected to reproduce.
# Agreement here is statistical rather than exact, because the host solves the
# implicit step with a finite-difference Jacobian and the kernel with an analytic
# one, so the Newton iterates differ within the solve tolerance.
function run_implicit_tau_kernel_tests(backend, nsims)
    leaping_prob(rates, rs, ns, u0, tspan) = JumpProblem(DiscreteProblem(u0, tspan),
        PureLeaping(), MassActionJump(rates, rs, ns))

    algs = (SimpleImplicitTauLeaping(), SimpleTrapezoidalLeaping())

    # Compare the mean of every species at every saved time against the serial
    # run of the same algorithm.
    #
    # The reference deliberately uses far fewer trajectories than the kernel. The
    # serial implicit solver builds a `NonlinearProblem` and runs a finite
    # difference Jacobian per step, so it is orders of magnitude slower than the
    # kernel here: on one of the models below, 20k trajectories take about 0.5 s
    # on the kernel and about 1000 s serially. A couple of thousand reference
    # trajectories pin the mean far tighter than the tolerances need.
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

    # Linear death X -> 0, for both formulations.
    for alg in algs
        jp = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [1000.0], (0.0, 4.0))
        sol = compare_means(jp, alg, nsims, 1; saveat = 1.0)
        @test sol.u[1].t == [0.0, 1.0, 2.0, 3.0, 4.0]
        @test sol.u[1].u[1] == [1000.0]
        @test all(issorted(reverse([sol.u[i].u[k][1] for k in 1:5])) for i in 1:100)
    end

    # Zero order 0 -> A: an empty reactant stoichiometry, so the Jacobian gets no
    # contribution from this reaction and stays the identity.
    let
        jp = leaping_prob([5.0], [Pair{Int, Int}[]], [[1 => 1]], [0.0], (0.0, 4.0))
        sol = compare_means(jp, SimpleImplicitTauLeaping(), nsims, 1; saveat = 2.0)
        for (k, t) in enumerate(sol.u[1].t)
            @test isapprox(mean(sol.u[i].u[k][1] for i in 1:nsims), 5.0 * t,
                rtol = 0.05, atol = 1.0e-8)
        end
    end

    # A genuinely stiff model: a fast reversible pair with a slow drain. This is
    # the case the implicit step exists for.
    for alg in algs
        jp = leaping_prob([100.0, 100.0, 0.1], [[1 => 1], [2 => 1], [1 => 1]],
            [[1 => -1, 2 => 1], [1 => 1, 2 => -1], [1 => -1, 3 => 1]],
            [1000.0, 1000.0, 0.0], (0.0, 1.0))
        sol = compare_means(jp, alg, nsims, 3; saveat = 0.25)
        # every reaction moves one molecule, so the total is conserved
        @test all(sum(sol.u[i].u[k]) == 2000.0 for i in 1:100, k in 1:5)
    end

    # SIR.
    for alg in algs
        jp = leaping_prob([0.1 / 1000, 0.01], [[1 => 1, 2 => 1], [2 => 1]],
            [[1 => -1, 2 => 1], [2 => -1, 3 => 1]], [999.0, 10.0, 0.0], (0.0, 100.0))
        compare_means(jp, alg, nsims, 3; saveat = 50.0)
    end

    # SEIR has four species, so the implicit step needs a 4x4 linear solve. That
    # is past the size where StaticArrays' own `\` is a closed form, which is why
    # the kernel carries its own non-throwing factorisation.
    for alg in algs
        jp = leaping_prob([0.3 / 1000, 0.2, 0.1],
            [[1 => 1, 3 => 1], [2 => 1], [3 => 1]],
            [[1 => -1, 2 => 1], [2 => -1, 3 => 1], [3 => -1, 4 => 1]],
            [999.0, 0.0, 10.0, 0.0], (0.0, 20.0))
        sol = compare_means(jp, alg, nsims, 4; saveat = 10.0)
        @test all(sum(sol.u[i].u[k]) == 1009.0 for i in 1:100, k in 1:3)
    end

    # Second order 2A -> B, exercising the falling factorial and its derivative.
    let
        jp = leaping_prob([0.01], [[1 => 2]], [[1 => -2, 2 => 1]], [200.0, 0.0],
            (0.0, 2.0))
        sol = compare_means(jp, SimpleImplicitTauLeaping(), nsims, 2; saveat = 1.0)
        @test all(all(iseven, (Int(sol.u[i].u[k][1]) for k in 1:3)) for i in 1:100)
    end

    # Third order 3A -> B, the deepest derivative branch.
    let
        jp = leaping_prob([1.0e-6], [[1 => 3]], [[1 => -3, 2 => 1]], [400.0, 0.0],
            (0.0, 5.0))
        compare_means(jp, SimpleImplicitTauLeaping(), nsims, 2; saveat = 2.5)
    end

    # Reversible binding A + B <-> C, with two conservation laws.
    let
        jp = leaping_prob([0.001, 0.5], [[1 => 1, 2 => 1], [3 => 1]],
            [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]],
            [1000.0, 800.0, 0.0], (0.0, 10.0))
        sol = compare_means(jp, SimpleTrapezoidalLeaping(), nsims, 3; saveat = 5.0)
        @test all(sol.u[i].u[k][1] + sol.u[i].u[k][3] == 1000.0 for i in 1:100, k in 1:3)
        @test all(sol.u[i].u[k][2] + sol.u[i].u[k][3] == 800.0 for i in 1:100, k in 1:3)
    end

    # Extinction: once nothing can fire the path is constant on the rest of the grid.
    let
        jp = leaping_prob([2.0], [[1 => 1]], [[1 => -1]], [3.0], (0.0, 20.0))
        sol = solve(EnsembleProblem(jp), SimpleImplicitTauLeaping(),
            EnsembleGPUKernel(backend); trajectories = 1000, saveat = 5.0)
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
            sol = solve(EnsembleProblem(jp), SimpleImplicitTauLeaping(; epsilon),
                EnsembleGPUKernel(backend); trajectories = nsims, saveat = tend)
            abs(mean(sol.u[i].u[end][1] for i in 1:nsims) - exact)
        end
        @test errs[1] > errs[2] > errs[3]
        @test errs[3] < 0.02 * exact
    end

    # saveat handling and the save_start / save_end flags.
    let
        jp = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [500.0], (0.0, 3.0))
        kern(; kwargs...) = solve(EnsembleProblem(jp), SimpleImplicitTauLeaping(),
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
    for alg in algs
        rj = RegularJump((out, u, p, t) -> (out[1] = 0.5u[1]),
            (du, u, p, t, counts, mark) -> (du[1] = -counts[1]), 1)
        rj_prob = JumpProblem(DiscreteProblem([50.0], (0.0, 3.0)), PureLeaping(), rj)
        @test_throws ErrorException solve(EnsembleProblem(rj_prob), alg,
            EnsembleGPUKernel(backend); trajectories = 10, saveat = 1.0)

        maj_prob = leaping_prob([0.5], [[1 => 1]], [[1 => -1]], [50.0], (0.0, 3.0))

        @test_throws ErrorException solve(EnsembleProblem(maj_prob), alg,
            EnsembleGPUKernel(backend); trajectories = 10)

        @test_throws ErrorException solve(EnsembleProblem(maj_prob), alg,
            EnsembleGPUKernel(backend); trajectories = 10, saveat = 1.0,
            callback = DiscreteCallback((u, t, integrator) -> false,
                integrator -> nothing))

        varying = EnsembleProblem(maj_prob; prob_func = (prob, ctx) -> prob)
        @test_throws ErrorException solve(varying, alg,
            EnsembleGPUKernel(backend); trajectories = 5, saveat = 1.0)
    end

    return nothing
end
