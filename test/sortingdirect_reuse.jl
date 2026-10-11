using JumpProcesses, SciMLBase, Test

function sortingdirect_reuse_problem(representation)
    prob = DiscreteProblem(zeros(Int, 3), (0.0, 2.0), [1.0, 2.0, 9.0])
    reactant_stoch = [Pair{Int, Int}[] for _ in 1:3]
    net_stoch = [[i => 1] for i in 1:3]
    if representation === :parameterized_massaction
        maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1, 2, 3])
        return JumpProblem(prob, SortingDirect(), maj)
    elseif representation === :fixed_massaction
        maj = MassActionJump(prob.p, reactant_stoch, net_stoch)
        return JumpProblem(prob, SortingDirect(), maj)
    end

    jumps = ntuple(3) do i
        rate(u, p, t) = p[i]
        function affect!(integrator)
            integrator.u[i] += 1
            nothing
        end
        ConstantRateJump(rate, affect!)
    end
    JumpProblem(prob, SortingDirect(), jumps...; dep_graph = [[i] for i in 1:3])
end

@testset "SortingDirect preserves learned search order" begin
    @testset "$representation" for representation in (:parameterized_massaction, :fixed_massaction, :constant)
        jprob = sortingdirect_reuse_problem(representation)
        sol = solve(jprob, SSAStepper(); seed = 12345)
        @test SciMLBase.successful_retcode(sol)
        @test sum(last(sol.u)) > 0

        # Seeded replay requires identical initial search order, as well as RNG
        # state. Each copy begins with the order learned by the completed solve.
        replay = solve(deepcopy(jprob), SSAStepper(); seed = 12345)
        repeated = solve(deepcopy(jprob), SSAStepper(); seed = 12345)
        @test replay.t == repeated.t
        @test replay.u == repeated.u

        # Initialization and rate resets retain the ordering optimization.
        aggregation = jprob.discrete_jump_aggregation
        aggregation.jump_search_order .= [3, 1, 2]
        integrator = init(jprob, SSAStepper(); seed = 12345)
        @test aggregation.jump_search_order == [3, 1, 2]
        if representation !== :fixed_massaction
            integrator.p .= [3.0, 4.0, 5.0]
        end
        reset_aggregated_jumps!(integrator)
        @test aggregation.jump_search_order == [3, 1, 2]
        @test aggregation.cur_rates ==
              (representation === :fixed_massaction ? [1.0, 2.0, 9.0] : [3.0, 4.0, 5.0])
        if representation === :parameterized_massaction
            @test aggregation.maj_rates == integrator.p
            @test jprob.massaction_jump.scaled_rates === nothing
        end
    end
end
