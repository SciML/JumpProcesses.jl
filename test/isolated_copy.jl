using JumpProcesses, OrdinaryDiffEq, SciMLBase, Test

# `isolated_copy` gives a problem its own jump state while sharing the wrapped problem.
# Copies must behave like independent problems (`deepcopy`), also when used at the same
# time as the original.

path(sol) = (sol.t, sol.u)

# A <-> B, as mass-action or constant-rate jumps
function ab_jprob(aggregator; massaction = true, kwargs...)
    prob = DiscreteProblem([12, 8], (0.0, 5.0), [0.7, 0.4])
    reactant_stoch = [[1 => 1], [2 => 1]]
    net_stoch = [[1 => -1, 2 => 1], [1 => 1, 2 => -1]]
    if massaction
        maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1, 2])
        return JumpProblem(prob, aggregator, maj; kwargs...)
    end

    ab!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    ba!(integrator) = (integrator.u[1] += 1; integrator.u[2] -= 1; nothing)
    jumps = (ConstantRateJump((u, p, t) -> p[1] * u[1], ab!),
        ConstantRateJump((u, p, t) -> p[2] * u[2], ba!))
    JumpProblem(prob, aggregator, jumps...; dep_graph = [[1, 2], [1, 2]],
        vartojumps_map = [[1], [2]], jumptovars_map = [[1, 2], [1, 2]], kwargs...)
end

# Steps two integrators in turn before finishing both, so that both are alive at once
function interleaved_paths(prob1, prob2, alg; seeds = (1, 2), nsteps = 10)
    integrators = (init(prob1, alg; seed = seeds[1]), init(prob2, alg; seed = seeds[2]))
    for _ in 1:nsteps, integrator in integrators
        step!(integrator)
    end
    foreach(solve!, integrators)
    map(integrator -> path(integrator.sol), integrators)
end

# Stops a simulation after `n` jumps; its counter is callback state
function stop_after_jumps(n)
    njumps = Ref(0)
    reset!(cb, u, t, integrator) = (njumps[] = 0)
    function affect!(integrator)
        njumps[] += 1
        njumps[] >= n && terminate!(integrator)
    end
    DiscreteCallback((u, t, integrator) -> true, affect!; initialize = reset!)
end

@testset "Shares the wrapped problem and copies the jump state" begin
    jprob = ab_jprob(Direct(); callback = stop_after_jumps(5), tstops = [1.0])
    jcopy = isolated_copy(jprob)

    @test jcopy isa typeof(jprob)
    @test jcopy.prob === jprob.prob
    @test jcopy.aggregator === jprob.aggregator
    @test jcopy.discrete_jump_aggregation !== jprob.discrete_jump_aggregation
    @test jcopy.jump_callback !== jprob.jump_callback
    @test jcopy.kwargs[:callback] !== jprob.kwargs[:callback]
    @test jcopy.kwargs[:tstops] == jprob.kwargs[:tstops]

    # The copied callback uses the copied aggregation, also in integrators
    jump_cb = only(jcopy.jump_callback.discrete_callbacks)
    @test jump_cb.condition === jcopy.discrete_jump_aggregation
    @test init(jcopy, SSAStepper()).cb.condition === jcopy.discrete_jump_aggregation

    # A remade copy shares the copy's jump state, not the original's
    remade = remake(jcopy; p = [1.0, 1.0])
    @test remade.discrete_jump_aggregation === jcopy.discrete_jump_aggregation
    @test remade.prob.p == [1.0, 1.0]
    @test jprob.prob.p == [0.7, 0.4]
end

@testset "Integrators alive at the same time: $(nameof(typeof(aggregator))), \
           massaction = $massaction" for aggregator in JumpProcesses.JUMP_AGGREGATORS,
    massaction in (true, false)

    jprob = ab_jprob(aggregator; massaction)
    refs = map(seed -> path(solve(deepcopy(jprob), SSAStepper(); seed)), (1, 2))

    # `jprob` stays unsolved until the last check: SortingDirect's learned order is copied
    copies = (isolated_copy(jprob), isolated_copy(jprob))
    @test interleaved_paths(copies..., SSAStepper()) == refs
    @test interleaved_paths(jprob, isolated_copy(jprob), SSAStepper()) == refs
end

@testset "ODE problems with variable-rate jumps: $(nameof(typeof(vr_aggregator)))" for vr_aggregator in (VR_FRM(), VR_Direct(), VR_DirectFW())

    # Growth of u[1], with births of u[2] at rate u[1] and resets of u[1] at a constant rate
    f!(du, u, p, t) = (du[1] = p[1] * u[1]; du[2] = 0; nothing)
    oprob = ODEProblem(f!, [1.0, 0.0], (0.0, 3.0), [0.5])
    birth = VariableRateJump((u, p, t) -> u[1],
        integrator -> (integrator.u[2] += 1; nothing))
    reset = ConstantRateJump((u, p, t) -> 1.0,
        integrator -> (integrator.u[1] = 1.0; nothing))
    jprob = JumpProblem(oprob, Direct(), birth, reset; vr_aggregator)

    refs = map(seed -> path(solve(deepcopy(jprob), Tsit5(); seed)), (1, 2))
    @test interleaved_paths(jprob, isolated_copy(jprob), Tsit5(); nsteps = 3) == refs
end

@testset "Concurrent solves" begin
    for jprob in (ab_jprob(SortingDirect()), ab_jprob(NRM(); massaction = false))
        refs = [path(solve(deepcopy(jprob), SSAStepper(); seed)) for seed in 1:8]
        tasks = [Threads.@spawn path(solve(isolated_copy(jprob), SSAStepper(); seed))
                 for seed in 1:8]
        @test fetch.(tasks) == refs
    end
end

@testset "Callbacks passed to the JumpProblem are copied" begin
    birth = MassActionJump([[0 => 1]], [[1 => 1]]; param_idxs = [1])
    jprob = JumpProblem(DiscreteProblem([0], (0.0, 100.0), [1.0]), Direct(), birth;
        callback = stop_after_jumps(5))

    # A shared counter would stop both simulations early
    (_, u1), (_, u2) = interleaved_paths(jprob, isolated_copy(jprob), SSAStepper();
        nsteps = 3)
    @test u1[end] == u2[end] == [5]
end

@testset "Copies don't depend on the original's memory" begin
    # Function-wrapper aggregators store the rates in wrappers that call through pointers
    function rate_seen_by_copy()
        k = 2.0
        rate = (u, p, t) -> k
        jprob = JumpProblem(DiscreteProblem([0], (0.0, 1.0)), DirectFW(),
            ConstantRateJump(rate, integrator -> (integrator.u[1] += 1; nothing)))
        isolated_copy(jprob)
    end
    copies = [rate_seen_by_copy() for _ in 1:20]
    for _ in 1:5
        GC.gc(true)
        junk = [fill(-1.0, 4) for _ in 1:10^4]
    end
    rate_at_start(jprob) = only(jprob.discrete_jump_aggregation.rates)(jprob.prob.u0,
        jprob.prob.p, 0.0)
    @test all(==(2.0) ∘ rate_at_start, copies)
end

@testset "Spatial aggregators: $(nameof(typeof(aggregator)))" for aggregator in (
    NSM(), DirectCRDirect())

    # A + B <--> C on a line of three sites
    majumps = MassActionJump([0.5, 1.0], [[1 => 1, 2 => 1], [3 => 1]],
        [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]])
    u0 = zeros(Int, 3, 3)
    u0[:, 2] .= [20, 20, 0]
    prob = DiscreteProblem(u0, (0.0, 2.0), [0.5, 1.0])
    jprob = JumpProblem(prob, aggregator, majumps; hopping_constants = fill(1.0, size(u0)),
        spatial_system = CartesianGrid((3,)))

    refs = map(seed -> path(solve(deepcopy(jprob), SSAStepper(); seed)), (1, 2))
    @test interleaved_paths(jprob, isolated_copy(jprob), SSAStepper()) == refs
end

@testset "PureLeaping" begin
    maj = MassActionJump([[1 => 1], [2 => 1]], [[1 => -1, 2 => 1], [1 => 1, 2 => -1]];
        param_idxs = [1, 2])
    jprob = JumpProblem(DiscreteProblem([12, 8], (0.0, 5.0), [0.7, 0.4]), PureLeaping(),
        maj)
    jcopy = isolated_copy(jprob)
    @test jcopy.prob === jprob.prob
    @test path(solve(jcopy, SimpleExplicitTauLeaping(); seed = 1)) ==
          path(solve(deepcopy(jprob), SimpleExplicitTauLeaping(); seed = 1))
end
