using JumpProcesses, SciMLBase, Test

# Small trajectory checks catch initialization and RNG plumbing failures without
# the ensembles used by the statistical correctness tests.
function check_ssa_smoke_solution(sol, prob)
    @test SciMLBase.successful_retcode(sol)
    @test first(sol.t) == first(prob.tspan)
    @test last(sol.t) == last(prob.tspan)
    @test issorted(sol.t)
    @test all(u -> all(>=(0), u), sol.u)
    @test all(u -> sum(u) == sum(prob.u0), sol.u)
end

function check_ssa_smoke_problem(jprob)
    # Independent trials start with identical mutable aggregator state. SortingDirect
    # intentionally retains its learned search order; see sortingdirect_reuse.jl.
    sol_default = solve(deepcopy(jprob), SSAStepper())
    check_ssa_smoke_solution(sol_default, jprob.prob)

    sol_seeded = solve(deepcopy(jprob), SSAStepper(); seed = 12345)
    check_ssa_smoke_solution(sol_seeded, jprob.prob)
    sol_repeated = solve(deepcopy(jprob), SSAStepper(); seed = 12345)
    @test sol_seeded.t == sol_repeated.t
    @test sol_seeded.u == sol_repeated.u
end

function ssa_smoke_problem(aggregator, representation)
    # A <-> B: equivalent mass-action and constant-rate representations.
    p = [0.7, 0.4]
    prob = DiscreteProblem([12, 8], (0.0, 0.5), p)
    reactant_stoch = [[1 => 1], [2 => 1]]
    net_stoch = [[1 => -1, 2 => 1], [1 => 1, 2 => -1]]

    if representation === :parameterized_massaction
        maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1, 2])
        return JumpProblem(prob, aggregator, maj)
    elseif representation === :fixed_massaction
        maj = MassActionJump(p, reactant_stoch, net_stoch)
        return JumpProblem(prob, aggregator, maj)
    end

    rate_ab(u, p, t) = p[1] * u[1]
    rate_b_to_a(u, p, t) = p[2] * u[2]
    function affect_ab!(integrator)
        integrator.u[1] -= 1
        integrator.u[2] += 1
        nothing
    end
    function affect_b_to_a!(integrator)
        integrator.u[1] += 1
        integrator.u[2] -= 1
        nothing
    end
    jumps = (ConstantRateJump(rate_ab, affect_ab!),
        ConstantRateJump(rate_b_to_a, affect_b_to_a!))
    dep_graph = [[1, 2], [1, 2]]
    vartojumps_map = [[1], [2]]
    jumptovars_map = [[1, 2], [1, 2]]
    JumpProblem(prob, aggregator, jumps...; dep_graph, vartojumps_map, jumptovars_map)
end

@testset "Nonspatial SSA aggregator smoke" begin
    @testset "$(typeof(aggregator)): $representation" for aggregator in JumpProcesses.JUMP_AGGREGATORS,
        representation in (:parameterized_massaction, :constant)

        check_ssa_smoke_problem(ssa_smoke_problem(aggregator, representation))
    end

    @testset "$(typeof(aggregator)): fixed mass action" for aggregator in (RSSA(), RSSACR())
        check_ssa_smoke_problem(ssa_smoke_problem(aggregator, :fixed_massaction))
    end
end

@testset "Automatic SSA selection smoke" begin
    # Duplicate A -> B channels keep the total rate fixed while crossing the
    # reaction-count thresholds for Direct, RSSA, and RSSACR.
    @testset "$num_reactions reactions" for (num_reactions, expected_aggregator) in ((
        19, Direct), (20, RSSA),
        (99, RSSA), (100, RSSACR))
        p = fill(0.7 / num_reactions, num_reactions)
        prob = DiscreteProblem([20, 0], (0.0, 0.5), p)
        reactant_stoch = [[1 => 1] for _ in 1:num_reactions]
        net_stoch = [[1 => -1, 2 => 1] for _ in 1:num_reactions]
        maj = MassActionJump(reactant_stoch, net_stoch;
            param_idxs = collect(1:num_reactions))
        jprob = JumpProblem(prob, JumpSet(maj))
        @test jprob.aggregator isa expected_aggregator
        check_ssa_smoke_problem(jprob)
    end
end

@testset "Spatial SSA aggregator smoke" begin
    @testset "$(typeof(aggregator))" for aggregator in (NSM(), DirectCRDirect())
        prob = DiscreteProblem([12 0; 0 8], (0.0, 0.5), [0.7, 0.4])
        maj = MassActionJump([[1 => 1], [2 => 1]],
            [[1 => -1, 2 => 1], [1 => 1, 2 => -1]]; param_idxs = [1, 2])
        spatial_system = CartesianGrid((2,))
        hopping_constants = [0.2, 0.2]
        jprob = JumpProblem(prob, aggregator, maj; spatial_system, hopping_constants)
        check_ssa_smoke_problem(jprob)
    end
end
