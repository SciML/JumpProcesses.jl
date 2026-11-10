using JumpProcesses, Test

@testset "Mass-action working rates follow current parameters" begin
    # 3A -> B has a 3! scaling factor, which must be applied exactly once.
    reactant_stoch = [[1 => 3]]
    net_stoch = [[1 => -3, 2 => 1]]
    @testset "$(typeof(algorithm))" for algorithm in JumpProcesses.JUMP_AGGREGATORS
        maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = 1)
        prob = DiscreteProblem([9, 0], (0.0, 0.1), [6.0])
        jprob = JumpProblem(prob, algorithm, maj)
        # A parameter change after construction must be seen by initialization.
        prob.p[1] = 12.0
        integrator = init(jprob, SSAStepper(); seed = 12345)
        @test jprob.discrete_jump_aggregation.maj_rates == [2.0]
        integrator.p[1] = 18.0
        reset_aggregated_jumps!(integrator)
        @test jprob.discrete_jump_aggregation.maj_rates == [3.0]
        @test jprob.massaction_jump.scaled_rates === nothing

        remade = remake(jprob; p = [24.0])
        init(remade, SSAStepper(); seed = 12345)
        @test remade.discrete_jump_aggregation.maj_rates == [4.0]
        @test jprob.prob.p == [18.0]
        @test remade.massaction_jump === jprob.massaction_jump
    end
end

@testset "Spatial mass-action working rates follow current parameters" begin
    @testset "$(typeof(algorithm))" for algorithm in (NSM(), DirectCRDirect())
        maj = MassActionJump([[1 => 3]], [[1 => -3, 2 => 1]]; param_idxs = 1)
        prob = DiscreteProblem([6 9; 0 0], (0.0, 0.1), [6.0])
        jprob = JumpProblem(prob, algorithm, maj;
            spatial_system = CartesianGrid((2,)), hopping_constants = zeros(2))
        prob.p[1] = 12.0
        integrator = init(jprob, SSAStepper(); seed = 12345)
        rx_rates = jprob.discrete_jump_aggregation.rx_rates
        @test rx_rates.maj_rates == [2.0]
        @test rx_rates.rates == [240.0 1008.0]
        integrator.p[1] = 18.0
        reset_aggregated_jumps!(integrator)
        @test rx_rates.maj_rates == [3.0]
        @test rx_rates.rates == [360.0 1512.0]
        @test jprob.massaction_jump.scaled_rates === nothing

        remade = remake(jprob; p = [24.0])
        init(remade, SSAStepper(); seed = 12345)
        @test remade.discrete_jump_aggregation.rx_rates.maj_rates == [4.0]
        @test remade.discrete_jump_aggregation.rx_rates.rates == [480.0 2016.0]
        @test jprob.prob.p == [18.0]
    end
end

@testset "Flattening materializes current scaled rates once" begin
    @testset "$parameterized" for parameterized in (false, true)
        reactant_stoch = [[1 => 3]]
        net_stoch = [[1 => -3, 2 => 1]]
        maj = parameterized ? MassActionJump(reactant_stoch, net_stoch; param_idxs = 1) :
              MassActionJump([12.0], reactant_stoch, net_stoch)
        prob = DiscreteProblem([6 9; 0 0], (0.0, 0.1), [6.0])
        prob.p[1] = 12.0
        jprob = JumpProblem(prob, Direct(), maj;
            spatial_system = CartesianGrid((2,)), hopping_constants = zeros(2, 2))
        @test jprob.massaction_jump.scaled_rates == [0.0, 0.0, 0.0, 0.0, 2.0, 2.0]
        @test jprob.prob.u0 == [6, 0, 9, 0]
        init(jprob, SSAStepper(); seed = 12345)
        @test jprob.discrete_jump_aggregation.maj_rates ==
              jprob.massaction_jump.scaled_rates
        if parameterized
            @test maj.scaled_rates === nothing
        else
            @test maj.scaled_rates == [2.0]
        end
    end
end

@testset "Mass-action merges preserve source arrays and parameter indices" begin
    @testset "$parameterized, $collection" for parameterized in (false, true),
        collection in (false, true)

        rs1, rs2 = [[1 => 3]], [[2 => 1]]
        ns1, ns2 = [[1 => -3, 2 => 1]], [[2 => -1, 1 => 3]]
        idxs1, idxs2 = [1], [2]
        maj1 = parameterized ? MassActionJump(rs1, ns1; param_idxs = idxs1) :
               MassActionJump([6.0], rs1, ns1)
        maj2 = parameterized ? MassActionJump(rs2, ns2; param_idxs = idxs2) :
               MassActionJump([2.0], rs2, ns2)
        jset = collection ? JumpSet(; massaction_jumps = [maj1, maj2]) : JumpSet(maj1, maj2)
        @test maj1.reactant_stoch == [[1 => 3]]
        @test maj2.reactant_stoch == [[2 => 1]]
        @test ns1 == [[1 => -3, 2 => 1]]
        @test ns2 == [[2 => -1, 1 => 3]]
        @test idxs1 == [1]
        @test idxs2 == [2]
        @test jset.massaction_jump.net_stoch == [only(ns1), only(ns2)]
        dest = zeros(2)
        JumpProcesses.fill_scaled_rates!(dest, jset.massaction_jump, [6.0, 2.0])
        @test dest == [1.0, 2.0]
        if parameterized
            @test maj1.param_mapper.param_idxs == [1]
            @test maj2.param_mapper.param_idxs == [2]
            @test jset.massaction_jump.param_mapper.param_idxs == [1, 2]
        else
            @test maj1.scaled_rates == [1.0]
            @test maj2.scaled_rates == [2.0]
        end
    end
end

# Scalar-form jumps hold one reaction without enclosing vectors.
@testset "Scalar mass-action jumps merge in any order" begin
    p = [2.0, 3.0, 5.0]
    fixed_scalar1() = MassActionJump(2.0, [1 => 1], [1 => -1, 2 => 1])
    fixed_scalar2() = MassActionJump(3.0, [2 => 2], [2 => -2, 3 => 1])  # scaled by 1/2!
    fixed_vector() = MassActionJump([5.0], [[3 => 1]], [[3 => -1]])
    param_scalar1() = MassActionJump([1 => 1], [1 => -1, 2 => 1]; param_idxs = 1)
    param_scalar2() = MassActionJump([2 => 2], [2 => -2, 3 => 1]; param_idxs = 2)
    param_vector() = MassActionJump([[3 => 1]], [[3 => -1]]; param_idxs = [3])
    cases = (("fixed scalar, fixed scalar", (fixed_scalar1, fixed_scalar2), [2.0, 1.5]),
        ("fixed scalar, fixed vector", (fixed_scalar1, fixed_vector), [2.0, 5.0]),
        ("fixed vector, fixed scalar", (fixed_vector, fixed_scalar1), [5.0, 2.0]),
        ("parameterized scalar, parameterized scalar", (param_scalar1, param_scalar2),
            [2.0, 1.5]),
        ("parameterized scalar, parameterized vector", (param_scalar1, param_vector),
            [2.0, 5.0]),
        ("parameterized vector, parameterized scalar", (param_vector, param_scalar1),
            [5.0, 2.0]),
        ("single fixed scalar", (fixed_scalar1,), [2.0]),
        ("single parameterized scalar", (param_scalar1,), [2.0]))
    dprob = DiscreteProblem([10, 10, 10], (0.0, 1.0), p)

    # A jump's reactions as a vector, whether the jump is in scalar or collection form.
    reactions(stoich) = eltype(stoich) <: Pair ? [stoich] : stoich

    @testset "$name" for (name, builders, rates) in cases
        jumps = map(build -> build(), builders)
        fresh_jumps = map(build -> build(), builders)
        # The merged jump lists each jump's reactions and parameter indices in order.
        expected_rs = reduce(vcat, map(maj -> reactions(maj.reactant_stoch), fresh_jumps))
        expected_ns = reduce(vcat, map(maj -> reactions(maj.net_stoch), fresh_jumps))
        parameterized = first(fresh_jumps).param_mapper !== nothing
        function check_merged(maj)
            @test maj.reactant_stoch == expected_rs
            @test maj.net_stoch == expected_ns
            if parameterized
                @test maj.param_mapper.param_idxs ==
                      reduce(vcat, map(m -> m.param_mapper.param_idxs, fresh_jumps))
            end
            dest = zeros(JumpProcesses.get_num_majumps(maj))
            JumpProcesses.fill_scaled_rates!(dest, maj, p)
            @test dest == rates
        end
        for jset in (JumpSet(jumps...), JumpSet(; massaction_jumps = collect(jumps)))
            check_merged(jset.massaction_jump)
        end
        jprob = JumpProblem(dprob, Direct(), jumps...)
        check_merged(jprob.massaction_jump)
        @test successful_retcode(solve(jprob, SSAStepper(); seed = 1))

        # Merging leaves the user's jumps unchanged.
        for (maj, fresh) in zip(jumps, fresh_jumps)
            @test maj.reactant_stoch == fresh.reactant_stoch
            @test maj.net_stoch == fresh.net_stoch
            @test maj.scaled_rates == fresh.scaled_rates
            if fresh.param_mapper !== nothing
                @test maj.param_mapper.param_idxs == fresh.param_mapper.param_idxs
            end
        end
    end

    @testset "Mixing fixed-rate and parameterized jumps is diagnosed" begin
        for builders in ((fixed_scalar1, param_scalar1), (param_scalar1, fixed_scalar1),
            (fixed_vector, param_vector), (param_vector, fixed_vector))
            jumps = map(build -> build(), builders)
            @test_throws r"fixed rates with one that has a parameter mapping" JumpSet(jumps...)
            @test_throws r"fixed rates with one that has a parameter mapping" JumpSet(;
                massaction_jumps = collect(jumps))
        end
    end
end

@testset "Removed reset rate-refresh keyword is diagnosed" begin
    maj = MassActionJump([[1 => 1]], [[1 => -1]]; param_idxs = 1)
    jprob = JumpProblem(DiscreteProblem([5], (0.0, 1.0), [1.0]), Direct(), maj)
    integrator = init(jprob, SSAStepper(); seed = 12345)
    for update_jump_params in (false, true)
        @test_throws ArgumentError reset_aggregated_jumps!(integrator; update_jump_params)
        @test_throws r"update_jump_params.*removed" reset_aggregated_jumps!(integrator;
            update_jump_params)
    end
end

@testset "Four-jump construction with a scalar mass-action jump" begin
    rate(u, p, t) = 1.0
    affect!(integrator) = nothing
    crj = ConstantRateJump(rate, affect!)
    for parameterized in (false, true)
        maj = parameterized ? MassActionJump([1 => 3], [1 => -3]; param_idxs = 1) :
              MassActionJump(6.0, [1 => 3], [1 => -3])
        jset = JumpSet(crj, crj, crj, maj)
        @test jset.constant_jumps == (crj, crj, crj)
        @test jset.variable_jumps == ()
        @test jset.regular_jump === nothing
        @test jset.massaction_jump.reactant_stoch == [[1 => 3]]
        dest = zeros(1)
        JumpProcesses.fill_scaled_rates!(dest, jset.massaction_jump, [6.0])
        @test dest == [1.0]
    end
end

@testset "Removed JumpProblem rate keywords are diagnosed" begin
    maj = MassActionJump([[1 => 3]], [[1 => -3]]; param_idxs = 1)
    prob = DiscreteProblem([6], (0.0, 1.0), [6.0])
    for algorithm in (Direct(), PureLeaping()), value in (false, true)

        @test_throws ArgumentError JumpProblem(prob, algorithm, maj; scale_rates = value)
        @test_throws r"scale_rates.*no longer.*MassActionJump" JumpProblem(prob, algorithm,
            maj; scale_rates = value)
        @test_throws ArgumentError JumpProblem(prob, algorithm, maj; useiszero = value)
        @test_throws r"useiszero.*no longer.*MassActionJump" JumpProblem(prob, algorithm,
            maj; useiszero = value)
    end
end
