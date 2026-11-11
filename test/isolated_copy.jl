using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, SciMLBase, Test
using JumpProcesses: shared_fields
using InteractiveUtils: subtypes

# `isolated_copy` gives a problem its own jump state and shares everything else. The tests of
# integrators alive at the same time, variable-rate jumps, threaded solves, spatial
# aggregators, independence from the original's memory and `PureLeaping` are adapted from
# those by Kaan Öcal in #711.

path(sol) = (sol.t, sol.u)

# A <-> B, as mass-action or constant-rate jumps, with state type `U` and time and parameter
# type `T`
function ab_jprob(aggregator; massaction = true, U = Int, T = Float64, kwargs...)
    prob = DiscreteProblem(U[12, 8], (zero(T), T(5)), T[0.7, 0.4])
    reactant_stoch = [[1 => 1], [2 => 1]]
    net_stoch = [[1 => -1, 2 => 1], [1 => 1, 2 => -1]]
    if massaction
        maj = MassActionJump(reactant_stoch, net_stoch; param_idxs = [1, 2])
        return JumpProblem(prob, aggregator, maj; kwargs...)
    end

    a_to_b!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    b_to_a!(integrator) = (integrator.u[1] += 1; integrator.u[2] -= 1; nothing)
    jumps = (ConstantRateJump((u, p, t) -> p[1] * u[1], a_to_b!),
        ConstantRateJump((u, p, t) -> p[2] * u[2], b_to_a!))
    dependencies = JumpDependencies(; dep_graph = [[1, 2], [1, 2]],
        vartojumps_map = [[1], [2]], jumptovars_map = [[1, 2], [1, 2]])
    JumpProblem(prob, aggregator, jumps...; dependencies, kwargs...)
end

# Steps two integrators in turn before finishing both, so that both are alive at once
function interleaved_paths(prob1, prob2, alg; seeds = (1, 2), nsteps = 10,
        kwargs1 = (;), kwargs2 = (;))
    integrators = (init(prob1, alg; seed = seeds[1], kwargs1...),
        init(prob2, alg; seed = seeds[2], kwargs2...))
    for _ in 1:nsteps
        foreach(step!, integrators)
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

# Growth of u[1], with births of u[2] at rate u[1] and resets of u[1] at a constant rate
function growth_jprob(vr_aggregator; T = Float64)
    f!(du, u, p, t) = (du[1] = p[1] * u[1]; du[2] = 0; nothing)
    oprob = ODEProblem(f!, T[1.0, 0.0], (zero(T), T(3)), T[0.5])
    birth = VariableRateJump((u, p, t) -> u[1],
        integrator -> (integrator.u[2] += 1; nothing))
    reset = ConstantRateJump((u, p, t) -> 1.0,
        integrator -> (integrator.u[1] = 1.0; nothing))
    JumpProblem(oprob, Direct(), birth, reset; vr_aggregator)
end

# The fields in `shared_fields` are the original's; every other field is a copy (numbers,
# which are never modified in place, are shared, and values of bits types are copied by
# value)
function test_copied_and_shared_fields(orig, copied)
    T = typeof(orig)
    @test typeof(copied) === T
    @test copied !== orig
    @test allunique(shared_fields(T))
    @test issubset(shared_fields(T), fieldnames(T))
    for name in fieldnames(T)
        x, y = getfield(orig, name), getfield(copied, name)
        if name in shared_fields(T)
            @test x === y
        else
            @test isbits(x) || x isa Number || x !== y
        end
    end
end

# Priority tables (`rt`, `ptt`) are copied down to their groups
function test_copied_tables(agg, agg_copy)
    for name in (:rt, :ptt)
        hasfield(typeof(agg), name) || continue
        table, table_copy = getfield(agg, name), getfield(agg_copy, name)
        test_copied_and_shared_fields(table, table_copy)
        @test length(table.groups) == length(table_copy.groups)
        for (group, group_copy) in zip(table.groups, table_copy.groups)
            test_copied_and_shared_fields(group, group_copy)
            @test group.pids == group_copy.pids
        end
    end
end

# The fields of the problem that every copy shares
const SHARED_PROBLEM_FIELDS = (:prob, :aggregator, :constant_jumps, :variable_jumps,
    :regular_jump, :massaction_jump, :kwargs)

@testset "Every aggregator declares the fields it shares" begin
    fallback = which(shared_fields, Tuple{Type{Nothing}})
    own(types) = filter(T -> parentmodule(T) === JumpProcesses, types)
    for T in own([subtypes(JumpProcesses.AbstractSSAJumpAggregator);
                  subtypes(JumpProcesses.AbstractHopRates)])
        @test which(shared_fields, Tuple{Type{T}}) !== fallback
        @test issubset(shared_fields(T), fieldnames(T))
    end

    # An object of a type that declares nothing is never shared by mistake
    @test_throws ArgumentError JumpProcesses.copy_working(Ref(1.0))
end

@testset "Shares the jumps and copies the jump state: $(nameof(typeof(aggregator))), \
           massaction = $massaction" for aggregator in JumpProcesses.JUMP_AGGREGATORS,
    massaction in (true, false)

    jprob = ab_jprob(aggregator; massaction, callback = stop_after_jumps(5), tstops = [1.0])
    jcopy = isolated_copy(jprob)

    @test jcopy isa typeof(jprob)
    for name in SHARED_PROBLEM_FIELDS
        @test getfield(jcopy, name) === getfield(jprob, name)
    end

    agg = jprob.discrete_jump_aggregation
    test_copied_and_shared_fields(agg, jcopy.discrete_jump_aggregation)
    test_copied_tables(agg, jcopy.discrete_jump_aggregation)

    # The copied callback uses the copied aggregation, also in integrators
    jump_cb = only(jcopy.jump_callback.discrete_callbacks)
    @test jump_cb.condition === jump_cb.affect! === jump_cb.initialize ===
          jcopy.discrete_jump_aggregation
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

@testset "Copies of a problem with a live integrator: $(nameof(typeof(aggregator)))" for aggregator in JumpProcesses.JUMP_AGGREGATORS
    jprob = ab_jprob(aggregator; massaction = false)
    fresh = isolated_copy(jprob)
    ref = path(solve(isolated_copy(jprob), SSAStepper(); seed = 1))

    integrator = init(jprob, SSAStepper(); seed = 1)
    for _ in 1:5
        step!(integrator)
    end
    live_copy = isolated_copy(jprob)
    if aggregator isa SortingDirect
        # the copy keeps the order learned so far
        @test live_copy.discrete_jump_aggregation.jump_search_order ==
              jprob.discrete_jump_aggregation.jump_search_order
    else
        @test path(solve(live_copy, SSAStepper(); seed = 2)) ==
              path(solve(fresh, SSAStepper(); seed = 2))
    end

    # A copy whose integrator has another type wraps the affects for that type, which must
    # leave the original integrator's wrappers alone
    other = init(isolated_copy(jprob), SSAStepper(); seed = 3, saveat = 0.5,
        callback = DiscreteCallback((u, t, integrator) -> false, integrator -> nothing))
    @test typeof(other) != typeof(integrator)
    solve!(other)

    # The original integrator continues as if no copy had been made
    solve!(integrator)
    @test path(integrator.sol) == ref
end

@testset "ODE problems with variable-rate jumps: $(nameof(typeof(vr_aggregator)))" for vr_aggregator in (VR_FRM(), VR_Direct(), VR_DirectFW())
    jprob = growth_jprob(vr_aggregator)
    jcopy = isolated_copy(jprob)

    @test jcopy isa typeof(jprob)
    for name in SHARED_PROBLEM_FIELDS
        @test getfield(jcopy, name) === getfield(jprob, name)
    end
    vr_cb = only(jprob.jump_callback.continuous_callbacks)
    vr_cb_copy = only(jcopy.jump_callback.continuous_callbacks)
    if vr_aggregator isa VR_FRM
        # its callbacks keep their state in the integrator's `u.jump_u`
        @test vr_cb_copy === vr_cb
    else
        cache, cache_copy = vr_cb.condition, vr_cb_copy.condition
        @test vr_cb_copy.affect! === cache_copy
        test_copied_and_shared_fields(cache, cache_copy)
        test_copied_and_shared_fields(cache.rate_fit, cache_copy.rate_fit)
        @test (vr_cb_copy.save_positions, vr_cb_copy.abstol, vr_cb_copy.reltol) ==
              (vr_cb.save_positions, vr_cb.abstol, vr_cb.reltol)
    end

    refs = map(seed -> path(solve(deepcopy(jprob), Tsit5(); seed)), (1, 2))
    @test interleaved_paths(jprob, isolated_copy(jprob), Tsit5(); nsteps = 3) == refs

    # A copy taken between the steps of a live integrator
    integrator = init(jprob, Tsit5(); seed = 1)
    foreach(_ -> step!(integrator), 1:3)
    @test path(solve(isolated_copy(jprob), Tsit5(); seed = 2)) == refs[2]
    solve!(integrator)
    @test path(integrator.sol) == refs[1]
end

@testset "Concurrent solves" begin
    for jprob in (ab_jprob(SortingDirect()), ab_jprob(NRM(); massaction = false))
        refs = [path(solve(deepcopy(jprob), SSAStepper(); seed)) for seed in 1:8]
        tasks = [Threads.@spawn path(solve(isolated_copy(jprob), SSAStepper(); seed))
                 for seed in 1:8]
        @test fetch.(tasks) == refs
    end

    # Jump-ODE problems, whose copies share the wrapped `ODEProblem`
    for vr_aggregator in (VR_FRM(), VR_Direct())
        jprob = growth_jprob(vr_aggregator)
        refs = [path(solve(deepcopy(jprob), Tsit5(); seed)) for seed in 1:8]
        tasks = [Threads.@spawn path(solve(isolated_copy(jprob), Tsit5(); seed))
                 for seed in 1:8]
        @test fetch.(tasks) == refs
    end
end

@testset "Callbacks passed to the JumpProblem are shared" begin
    birth = MassActionJump([[0 => 1]], [[1 => 1]]; param_idxs = [1])
    dprob = DiscreteProblem([0], (0.0, 100.0), [1.0])
    jprob = JumpProblem(dprob, Direct(), birth; callback = stop_after_jumps(5))
    @test isolated_copy(jprob).kwargs[:callback] === jprob.kwargs[:callback]

    # Callbacks with state passed to `init` are not, so each simulation stops at its own
    # fifth jump
    jprob = JumpProblem(dprob, Direct(), birth)
    (_, u1), (_, u2) = interleaved_paths(jprob, isolated_copy(jprob), SSAStepper();
        nsteps = 3, kwargs1 = (; callback = stop_after_jumps(5)),
        kwargs2 = (; callback = stop_after_jumps(5)))
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
        junk = [Ref(-1.0) for _ in 1:200_000]
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
    jcopy = isolated_copy(jprob)

    agg = jprob.discrete_jump_aggregation
    test_copied_and_shared_fields(agg, jcopy.discrete_jump_aggregation)
    test_copied_and_shared_fields(agg.rx_rates, jcopy.discrete_jump_aggregation.rx_rates)
    test_copied_and_shared_fields(agg.hop_rates, jcopy.discrete_jump_aggregation.hop_rates)
    test_copied_tables(agg, jcopy.discrete_jump_aggregation)

    refs = map(seed -> path(solve(deepcopy(jprob), SSAStepper(); seed)), (1, 2))
    @test interleaved_paths(jprob, isolated_copy(jprob), SSAStepper()) == refs
end

# A fresh aggregator's buffers are allocated with `undef`, so for `BigFloat` and `BigInt` they
# hold unassigned entries until the first `init`; these problems are copied unsolved.
# (`RDirect` can't build problems with `BigFloat` times yet.)
@testset "Copies of unsolved problems with BigFloat times: $(nameof(typeof(aggregator))), \
           massaction = $massaction" for aggregator in (Direct(), DirectFW(), DirectCR(),
        SortingDirect(), RSSA(), RSSACR(), FRM(), FRMFW(), NRM(), Coevolve(), CCNRM()),
    massaction in (true, false)

    jprob = ab_jprob(aggregator; massaction, T = BigFloat)
    jcopy = isolated_copy(jprob)
    test_copied_and_shared_fields(jprob.discrete_jump_aggregation,
        jcopy.discrete_jump_aggregation)
    @test path(solve(jcopy, SSAStepper(); seed = 1)) ==
          path(solve(deepcopy(jprob), SSAStepper(); seed = 1))
end

@testset "Copies of unsolved problems with BigInt states and BigFloat ODEs" begin
    jprob = ab_jprob(RSSA(); U = BigInt)
    jcopy = isolated_copy(jprob)
    @test jcopy.discrete_jump_aggregation.ulow !== jprob.discrete_jump_aggregation.ulow
    @test path(solve(jcopy, SSAStepper(); seed = 1)) ==
          path(solve(deepcopy(jprob), SSAStepper(); seed = 1))

    for vr_aggregator in (VR_Direct(), VR_DirectFW())
        jprob = growth_jprob(vr_aggregator; T = BigFloat)
        jcopy = isolated_copy(jprob)
        cache = only(jprob.jump_callback.continuous_callbacks).condition
        cache_copy = only(jcopy.jump_callback.continuous_callbacks).condition
        test_copied_and_shared_fields(cache, cache_copy)
        test_copied_and_shared_fields(cache.rate_fit, cache_copy.rate_fit)
        @test path(solve(jcopy, Tsit5(); seed = 1)) ==
              path(solve(deepcopy(jprob), Tsit5(); seed = 1))
    end
end

@testset "Jump-diffusions, through `resetted_jump_problem`" begin
    f!(du, u, p, t) = (du .= 0; nothing)
    g!(du, u, p, t) = (du .= 0.1; nothing)
    sprob = SDEProblem(f!, g!, [10.0, 5.0], (0.0, 2.0), [0.7, 0.4])
    a_to_b!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    b_to_a!(integrator) = (integrator.u[1] += 1; integrator.u[2] -= 1; nothing)
    jprob = JumpProblem(sprob, Direct(),
        ConstantRateJump((u, p, t) -> p[1] * max(u[1], 0), a_to_b!),
        ConstantRateJump((u, p, t) -> p[2] * max(u[2], 0), b_to_a!))

    jcopy = JumpProcesses.resetted_jump_problem(jprob, nothing)
    @test jcopy.prob === jprob.prob
    @test jcopy.discrete_jump_aggregation !== jprob.discrete_jump_aggregation

    # `alias_jumps = false` solves a copy, leaving the problem's jump state alone
    rates = copy(jprob.discrete_jump_aggregation.cur_rates)
    sol_copy = solve(jprob, SRIW1(); seed = 1,
        alias = SciMLBase.SDEAliasSpecifier(alias_jumps = false))
    @test isequal(jprob.discrete_jump_aggregation.cur_rates, rates)
    sol = solve(jprob, SRIW1(); seed = 1,
        alias = SciMLBase.SDEAliasSpecifier(alias_jumps = true))
    @test path(sol_copy) == path(sol)
end

@testset "With `alias_u0 = true`, a copy starts from the integrator's current state" begin
    jprob = ab_jprob(Direct())
    integrator = init(jprob, SSAStepper(); seed = 1,
        alias = SciMLBase.DiscreteAliasSpecifier(alias_u0 = true))
    foreach(_ -> step!(integrator), 1:5)
    @test integrator.u === jprob.prob.u0
    @test init(isolated_copy(jprob), SSAStepper(); seed = 2).u == integrator.u
end

@testset "PureLeaping" begin
    maj = MassActionJump([[1 => 1], [2 => 1]], [[1 => -1, 2 => 1], [1 => 1, 2 => -1]];
        param_idxs = [1, 2])
    jprob = JumpProblem(DiscreteProblem([12, 8], (0.0, 5.0), [0.7, 0.4]), PureLeaping(),
        maj)
    jcopy = isolated_copy(jprob)
    @test jcopy !== jprob
    @test all(name -> getfield(jcopy, name) === getfield(jprob, name),
        fieldnames(typeof(jprob)))
    @test path(solve(jcopy, SimpleExplicitTauLeaping(); seed = 1)) ==
          path(solve(deepcopy(jprob), SimpleExplicitTauLeaping(); seed = 1))
end
