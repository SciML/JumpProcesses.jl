using JumpProcesses, Test, StableRNGs, FunctionWrappers
rng = StableRNG(12345)

# https://github.com/SciML/JumpProcesses.jl/issues/324
let
    rate(u, p, t; debug = true) = 5.0
    function affect!(integrator)
        integrator.u[1] += 1
        nothing
    end
    jump = VariableRateJump(rate, affect!; urate = (u, p, t) -> 10.0,
        rateinterval = (u, p, t) -> 0.1)

    prob = DiscreteProblem([0.0], (0.0, 2.0), [1.0])
    jprob = JumpProblem(prob, Coevolve(), jump; variablerate_aggregator=NextReactionODE(), dep_graph = [[1]], rng)
    agg = jprob.discrete_jump_aggregation
    @test agg.affects! isa Vector{Any}

    integ = init(jprob, SSAStepper())
    T = Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{typeof(integ)}}}
    @test agg.affects! isa T
    affs = agg.affects!
    sol_c = solve!(integ)

    # check the affects vector is unchanged from a second call
    integ = init(jprob, SSAStepper())
    sol_c = solve!(integ)
    @test affs === agg.affects!

    # check changing the integrator doesn't break things
    terminate_condition(u, t, integrator) = (return u[1] >= 1)
    terminate_affect!(integrator) = terminate!(integrator)
    terminate_cb = DiscreteCallback(terminate_condition, terminate_affect!)
    integ2 = init(jprob, SSAStepper(); callback = terminate_cb)
    T2 = Vector{FunctionWrappers.FunctionWrapper{Nothing, Tuple{typeof(integ2)}}}
    @test T2 !== T
    @test agg.affects! isa T2

    # test the affects! Vector was recreated
    @test affs !== agg.affects!
    affs2 = agg.affects!
    solve!(integ2)

    # check affs2 is unchanged when solving again now
    integ2 = init(jprob, SSAStepper(); callback = terminate_cb)
    solve!(integ2)
    @test affs2 === agg.affects!
end
