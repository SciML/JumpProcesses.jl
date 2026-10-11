using JumpProcesses, DiffEqCallbacks, SciMLBase, StaticArrays, Test

# Evaluating an `SSAIntegrator` at a time: only `integrator.t` by default, and anywhere in
# the current step, `[integrator.tprev, integrator.t]`, with `SSAStepper(; save_uprev =
# true)`, which keeps the state at the start of each step in `integrator.uprev`.

errmsg(f) =
    try
        f()
        ""
    catch err
        sprint(showerror, err)
    end

mutable_state(u) = !(u isa Union{Number, SVector})

# Birth–death models built from constant-rate jumps with user-written effects, for vector
# and scalar states.
const VECTOR_BIRTH = ConstantRateJump((u, p, t) -> 2.0,
    integrator -> (integrator.u[1] += 1; nothing))
const VECTOR_DEATH = ConstantRateJump((u, p, t) -> 0.1 * u[1],
    integrator -> (integrator.u[1] -= 1; nothing))
const SCALAR_BIRTH = ConstantRateJump((u, p, t) -> 2.0,
    integrator -> (integrator.u += 1; nothing))
const SCALAR_DEATH = ConstantRateJump((u, p, t) -> 0.1 * u,
    integrator -> (integrator.u -= 1; nothing))

function vector_jprob(; u0 = [0], tspan = (0.0, 20.0), kwargs...)
    JumpProblem(DiscreteProblem(u0, tspan), Direct(), VECTOR_BIRTH, VECTOR_DEATH; kwargs...)
end

@testset "SSAStepper type and traits" begin
    @test SSAStepper() === SSAStepper{false}()
    @test SSAStepper(; save_uprev = true) === SSAStepper{true}()
    @test_throws ArgumentError SSAStepper{1}()
    jprob = vector_jprob()
    for alg in (SSAStepper(), SSAStepper(; save_uprev = true))
        @test alg isa SSAStepper
        @test SciMLBase.allows_late_binding_tstops(alg)
        @test SciMLBase.supports_solve_rng(jprob, alg)
    end
    # Solving without an algorithm still selects the default `SSAStepper()`.
    @test solve(jprob; seed = 1).alg === SSAStepper()
end

@testset "Default SSAStepper only evaluates at the current time" begin
    integrator = init(vector_jprob(), SSAStepper(); seed = 1)
    @test integrator.uprev === nothing
    @test integrator(integrator.t) == integrator.u
    step!(integrator)
    step!(integrator)
    current = integrator(integrator.t)
    @test current == integrator.u
    @test current !== integrator.u
    out = zeros(1)
    @test integrator(out, integrator.t) === out
    @test out == integrator.u

    @test_throws ArgumentError integrator(integrator.tprev)
    @test_throws ArgumentError integrator(out, integrator.tprev)
    msg = errmsg(() -> integrator(integrator.tprev))
    @test occursin("save_uprev = true", msg)
    @test occursin("tstops", msg)
    @test_throws ArgumentError SciMLBase.get_tmp_cache(integrator)
    @test occursin("save_uprev = true", errmsg(() -> SciMLBase.get_tmp_cache(integrator)))
end

@testset "SavingCallback is exact with save_uprev or tstops" begin
    ts = collect(0.0:0.5:20.0)
    # A pure-death model goes extinct early, so many save times fall after its last
    # event and are evaluated during `solve!`'s final advance to the end time.
    death_only = ConstantRateJump((u, p, t) -> 1.0 * u[1],
        integrator -> (integrator.u[1] -= 1; nothing))
    models = (("vector birth–death", [0], (VECTOR_BIRTH, VECTOR_DEATH)),
        ("scalar birth–death", 0, (SCALAR_BIRTH, SCALAR_DEATH)),
        ("vector pure death", [5], (death_only,)))
    for (name, u0, jumps) in models, save_positions in ((true, true), (false, false))

        @testset "$name, save_positions = $save_positions" begin
            dprob = DiscreteProblem(u0, (0.0, 20.0))
            jprob = JumpProblem(dprob, Direct(), jumps...; save_positions)
            # Native `saveat` is exact and runs before jumps; the seeded path is the same
            # for every saving choice.
            reference_prob = JumpProblem(dprob, Direct(), jumps...;
                save_positions = (false, false))
            reference = [first(u)
                         for u in solve(reference_prob, SSAStepper(); saveat = ts,
                seed = 7).u]
            saved_values() = SavedValues(Float64, typeof(first(u0)))
            saving_callback(values) = SavingCallback((u, t, integrator) -> first(u),
                values; saveat = ts)

            values = saved_values()
            solve(jprob, SSAStepper(; save_uprev = true);
                callback = saving_callback(values), seed = 7)
            @test values.t == ts
            @test values.saveval == reference

            # Stopping at the save times avoids evaluating earlier times altogether.
            values = saved_values()
            solve(jprob, SSAStepper(); callback = saving_callback(values), tstops = ts,
                seed = 7)
            @test values.t == ts
            @test values.saveval == reference

            @test_throws ArgumentError solve(jprob, SSAStepper();
                callback = saving_callback(saved_values()), seed = 7)
        end
    end
end

@testset "uprev is the state after the previous step's callbacks" begin
    # A callback that resets `u` without saving leaves the reset state out of the saved
    # solution, but `uprev` still records it.
    birth = ConstantRateJump((u, p, t) -> 1.0,
        integrator -> (integrator.u[1] += 1; nothing))
    reset = DiscreteCallback((u, t, integrator) -> u[1] == 3,
        integrator -> (integrator.u[1] = 0; nothing); save_positions = (false, false))
    jprob = JumpProblem(DiscreteProblem([0], (0.0, 100.0)), Direct(), birth)
    integrator = init(jprob, SSAStepper(; save_uprev = true); callback = reset, seed = 3)
    while integrator.u[1] != 0 || integrator.t == 0
        step!(integrator)
    end
    reset_time = integrator.t
    @test integrator.sol.u[end] == [3]
    step!(integrator)
    @test integrator.tprev == reset_time
    @test integrator(integrator.tprev) == [0]
    @test integrator((integrator.tprev + integrator.t) / 2) == [0]
end

@testset "Changing u between manual steps" begin
    integrator = init(vector_jprob(), SSAStepper(; save_uprev = true); seed = 1)
    step!(integrator)
    step!(integrator)
    previous = copy(integrator.uprev)
    integrator.u[1] = 100
    reset_aggregated_jumps!(integrator)
    @test integrator(integrator.t) == [100]
    # The state on `[tprev, t)` is unchanged by the modification at `t`.
    @test integrator(integrator.tprev) == previous
    step!(integrator)
    @test integrator(integrator.tprev) == [100]
end

@testset "End of solve" begin
    for save_uprev in (false, true)
        integrator = init(vector_jprob(), SSAStepper(; save_uprev); seed = 5)
        solve!(integrator)
        @test integrator.t == 20.0
        # The final advance to the end time starts a step at the last event.
        @test integrator.tprev == integrator.sol.t[end - 1]
        @test integrator.tprev < 20.0
        if save_uprev
            for t in (integrator.tprev, (integrator.tprev + 20.0) / 2, 20.0)
                @test integrator(t) == integrator.u
            end
        else
            @test_throws ArgumentError integrator(integrator.tprev)
        end
    end

    # A final callback that changes `u` at the end time: earlier times in the last step
    # keep the state after the last event.
    jprob = vector_jprob(; save_positions = (false, false))
    after_last_event = solve(jprob, SSAStepper(); seed = 5).u[end]
    final_change = DiscreteCallback((u, t, integrator) -> t == 20.0,
        integrator -> (integrator.u[1] = 1000; nothing))
    integrator = init(jprob, SSAStepper(; save_uprev = true); callback = final_change,
        seed = 5)
    solve!(integrator)
    @test integrator(20.0) == [1000]
    @test integrator.uprev == after_last_event
    @test integrator(integrator.tprev) == after_last_event
    @test integrator((integrator.tprev + 20.0) / 2) == after_last_event
end

@testset "Terminated solves do not advance" begin
    birth = ConstantRateJump((u, p, t) -> 2.0,
        integrator -> (integrator.u[1] += 1; nothing))
    stop = DiscreteCallback((u, t, integrator) -> u[1] >= 3, terminate!)
    jprob = JumpProblem(DiscreteProblem([0], (0.0, 20.0)), Direct(), birth)
    integrator = init(jprob, SSAStepper(; save_uprev = true); callback = stop, seed = 2)
    solve!(integrator)
    @test integrator.sol.retcode == ReturnCode.Terminated
    @test integrator.t < 20.0
    @test integrator(integrator.t) == integrator.u
    @test integrator(integrator.tprev) == integrator.uprev
    @test_throws ArgumentError integrator((integrator.t + 20.0) / 2)
end

@testset "isdenseplot for both SSAStepper settings" begin
    for alg in (SSAStepper(), SSAStepper(; save_uprev = true))
        # SciMLBase's generic method returns `true` for every `DiscreteProblem`.
        sparse = solve(vector_jprob(; save_positions = (false, false)), alg; saveat = 1.0,
            seed = 1)
        @test !SciMLBase.isdenseplot(sparse)
        @test SciMLBase.isdenseplot(solve(vector_jprob(), alg; seed = 1))
    end
end

# Supported state types: vectors and `SVector`s of integers or floats (here with
# mass-action jumps) and species × sites matrices for the spatial solvers.
function birth_death_maj_jprob(u0; tspan = (0.0, 20.0), kwargs...)
    maj = MassActionJump([[0 => 1], [1 => 1]], [[1 => 1], [1 => -1]];
        param_idxs = [1, 2])
    JumpProblem(DiscreteProblem(u0, tspan, [2.0, 0.1]), Direct(), maj; kwargs...)
end

function spatial_jprob(aggregator; tspan = (0.0, 20.0), kwargs...)
    u0 = zeros(Int, 1, 3)
    u0[1, 2] = 10
    JumpProblem(DiscreteProblem(u0, tspan), aggregator, JumpSet(nothing);
        hopping_constants = ones(1, 3), spatial_system = CartesianGridRej((3,)), kwargs...)
end

const STATE_CASES = (("Vector{Int}", kw -> birth_death_maj_jprob([10]; kw...)),
    ("Vector{Float64}", kw -> birth_death_maj_jprob([10.0]; kw...)),
    ("SVector{Int}", kw -> birth_death_maj_jprob(SA[10]; kw...)),
    ("SVector{Float64}", kw -> birth_death_maj_jprob(SA[10.0]; kw...)),
    ("Matrix{Int} with NSM", kw -> spatial_jprob(NSM(); kw...)),
    ("Matrix{Int} with DirectCRDirect", kw -> spatial_jprob(DirectCRDirect(); kw...)))

function check_current_evaluation(integrator)
    u = integrator.u
    value = integrator(integrator.t)
    @test value == u
    @test value isa typeof(u)
    if mutable_state(u)
        @test value !== u
        state = copy(u)
        value .= -1
        @test integrator.u == state
    end
    out = zeros(Float64, size(u))
    @test integrator(out, integrator.t) === out
    @test out == u
end

function check_step_evaluation(integrator, save_uprev)
    step!(integrator)
    before = copy(integrator.u)
    step!(integrator)
    @test integrator.u != before
    check_current_evaluation(integrator)
    middle = (integrator.tprev + integrator.t) / 2
    if save_uprev
        for t in (integrator.tprev, middle)
            value = integrator(t)
            @test value == before
            @test value isa typeof(integrator.u)
            if mutable_state(integrator.u)
                @test value !== integrator.uprev
                @test value !== integrator.u
                @test value !== only(SciMLBase.get_tmp_cache(integrator))
                value .= -1
                @test integrator(t) == before
            end
            out = zeros(Float64, size(integrator.u))
            @test integrator(out, t) === out
            @test out == before
        end
        cache = SciMLBase.get_tmp_cache(integrator)
        if mutable_state(integrator.u)
            @test cache isa Tuple{typeof(integrator.u)}
            @test only(cache) === only(SciMLBase.get_tmp_cache(integrator))
            @test only(cache) !== integrator.uprev
            @test only(cache) !== integrator.u
            @test integrator.uprev !== integrator.u
        else
            @test cache === nothing
        end
    else
        @test integrator.uprev === nothing
        @test_throws ArgumentError integrator(middle)
        @test_throws ArgumentError integrator(zeros(Float64, size(integrator.u)), middle)
        @test_throws ArgumentError SciMLBase.get_tmp_cache(integrator)
    end
end

function check_alias(make_jprob, save_uprev)
    for alias_u0 in (nothing, true, false)
        jprob = make_jprob((;))
        u0 = copy(jprob.prob.u0)
        alias = alias_u0 === nothing ? nothing :
                SciMLBase.DiscreteAliasSpecifier(; alias_u0)
        integrator = init(jprob, SSAStepper(; save_uprev); seed = 1, alias)
        if mutable_state(u0)
            @test (integrator.u === jprob.prob.u0) == (alias_u0 === true)
            if save_uprev
                @test integrator.uprev !== integrator.u
                @test integrator.uprev !== jprob.prob.u0
            end
        else
            @test integrator.u == jprob.prob.u0
        end
        step!(integrator)
        step!(integrator)
        alias_u0 === true || @test jprob.prob.u0 == u0
    end
end

function check_saving(make_jprob, save_uprev)
    jprob = make_jprob((; save_positions = (false, false)))
    u0 = copy(jprob.prob.u0)
    integrator = init(jprob, SSAStepper(; save_uprev); seed = 1, saveat = 1.0)
    solve!(integrator)
    sol = integrator.sol
    @test length(sol.t) == 21
    @test eltype(sol.u) == typeof(integrator.u)
    @test sol.u[1] == u0
    if mutable_state(integrator.u)
        @test allunique(objectid.(sol.u))
        @test all(saved -> saved !== integrator.u, sol.u)
    end
end

@testset "Supported state type: $name" for (name, make_jprob) in STATE_CASES
    for save_uprev in (false, true)
        integrator = init(make_jprob((;)), SSAStepper(; save_uprev); seed = 1)
        check_current_evaluation(integrator)
        check_step_evaluation(integrator, save_uprev)
        check_alias(make_jprob, save_uprev)
        check_saving(make_jprob, save_uprev)
    end
end

@testset "Scalar states in models without mass-action jumps" begin
    graphs = (; dep_graph = [[1, 2], [1, 2]], vartojumps_map = [[2]],
        jumptovars_map = [[1], [1]])
    aggregators = filter(agg -> !(agg isa Union{RSSA, RSSACR}),
        JumpProcesses.JUMP_AGGREGATORS)
    bounded_birth = VariableRateJump((u, p, t) -> 1.0 + 0.1 * sin(t),
        integrator -> (integrator.u += 1; nothing); urate = (u, p, t) -> 1.1,
        rateinterval = (u, p, t) -> Inf)
    for u0 in (0, 0.0)
        for aggregator in aggregators
            jprob = JumpProblem(DiscreteProblem(u0, (0.0, 20.0)), aggregator, SCALAR_BIRTH,
                SCALAR_DEATH; save_positions = (false, false), graphs...)
            sol = solve(jprob, SSAStepper(); saveat = 1.0, seed = 1)
            @test length(sol.u) == 21
            @test eltype(sol.u) == typeof(u0)
        end
        jprob = JumpProblem(DiscreteProblem(u0, (0.0, 20.0)), Coevolve(), bounded_birth;
            dep_graph = [[1]])
        sol = solve(jprob, SSAStepper(); seed = 1)
        @test eltype(sol.u) == typeof(u0)
        @test sol.u[end] > u0

        jprob = JumpProblem(DiscreteProblem(u0, (0.0, 20.0)), Direct(), SCALAR_BIRTH,
            SCALAR_DEATH)
        for save_uprev in (false, true)
            integrator = init(jprob, SSAStepper(; save_uprev); seed = 1)
            step!(integrator)
            before = integrator.u
            step!(integrator)
            @test integrator(integrator.t) == integrator.u
            @test integrator(integrator.t) isa typeof(u0)
            out = zeros(1)
            @test integrator(out, integrator.t) === out
            @test out == [integrator.u]
            middle = (integrator.tprev + integrator.t) / 2
            if save_uprev
                @test integrator(middle) == before
                @test integrator(middle) isa typeof(u0)
                @test integrator(out, middle) === out
                @test out == [before]
                @test SciMLBase.get_tmp_cache(integrator) === nothing
            else
                @test_throws ArgumentError integrator(middle)
            end
            for alias in (nothing, true, false)
                @test SciMLBase.successful_retcode(solve(jprob,
                    SSAStepper(; save_uprev); seed = 1, alias))
            end
        end
    end
end

function allocations_per_step(integrator)
    step!(integrator)
    step!(integrator)
    return @allocated step!(integrator)
end

@testset "Stepping with save_uprev does not allocate" begin
    # Measured with solution saving off, since saved snapshots allocate by design.
    for (name, make_jprob) in STATE_CASES
        name in ("Vector{Int}", "SVector{Int}", "Matrix{Int} with NSM") || continue
        jprob = make_jprob((; tspan = (0.0, 1.0e6), save_positions = (false, false)))
        integrator = init(jprob, SSAStepper(; save_uprev = true); seed = 1,
            save_start = false, save_end = false)
        @test allocations_per_step(integrator) == 0
    end
end

@testset "Evaluation and stepping infer" begin
    for save_uprev in (false, true)
        integrator = init(birth_death_maj_jprob([10]), SSAStepper(; save_uprev); seed = 1)
        step!(integrator)
        @inferred step!(integrator)
        @inferred integrator(integrator.t)
        @inferred integrator(zeros(1), integrator.t)
        if save_uprev
            @inferred integrator(integrator.tprev)
            @inferred SciMLBase.get_tmp_cache(integrator)
        end
    end
end

@testset "Integrators and ensembles own their uprev buffers" begin
    jprob = birth_death_maj_jprob([10])
    first_integrator = init(jprob, SSAStepper(; save_uprev = true); seed = 1)
    second_integrator = init(deepcopy(jprob), SSAStepper(; save_uprev = true); seed = 2)
    @test first_integrator.uprev !== second_integrator.uprev
    @test only(SciMLBase.get_tmp_cache(first_integrator)) !==
          only(SciMLBase.get_tmp_cache(second_integrator))

    paths(sol) = [(trajectory.t, trajectory.u) for trajectory in sol.u]
    run(seed) = solve(EnsembleProblem(jprob), SSAStepper(; save_uprev = true),
        EnsembleThreads(); trajectories = 16, seed)
    @test all(SciMLBase.successful_retcode, run(4).u)
    @test paths(run(4)) == paths(run(4))
end
