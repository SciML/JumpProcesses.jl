using JumpProcesses, OrdinaryDiffEq, SciMLBase, Test
using StaticArrays: SVector

# Ownership rules for JumpProcesses v10: JumpProcesses-owned solvers reuse (and
# re-initialize) the problem's jump state instead of copying it, `alias_jump` is
# removed, and SSAStepper supports the common `alias` controls for `u0`, `p` and
# `tstops`.

function bd_maj()
    MassActionJump([[0 => 1], [1 => 1]], [[1 => 1], [1 => -1]]; param_idxs = [1, 2])
end

function bd_jprob(; aggregator = Direct(), u0 = [10], p = [1.0, 0.1],
        tspan = (0.0, 10.0), kwargs...)
    JumpProblem(DiscreteProblem(u0, tspan, p), aggregator, bd_maj(); kwargs...)
end

const COUPLED_JUMP = ConstantRateJump((u, p, t) -> 1.0,
    integrator -> (integrator.u[1] += 1; nothing))

function ode_jprob()
    oprob = ODEProblem((du, u, p, t) -> (du .= 0; nothing), [10.0], (0.0, 1.0))
    JumpProblem(oprob, Direct(), COUPLED_JUMP)
end

function uses_problem_callback(integrator, jprob)
    integrator.cb.condition === jprob.discrete_jump_aggregation
end
function uses_problem_aggregation(integrator, jprob)
    any(cb -> cb.condition === jprob.discrete_jump_aggregation,
        integrator.opts.callback.discrete_callbacks)
end

const REMOVED_MSG = "`alias_jump` keyword argument was removed in JumpProcesses v10"

# The OrdinaryDiffEq umbrella loads, but does not export, its DAE solvers.
function loaded_module(name)
    only(filter(m -> nameof(m) === name, collect(values(Base.loaded_modules))))
end
const DFBDF = loaded_module(:OrdinaryDiffEqBDF).DFBDF

# Saved states of the nested test model are independent when no two share an inner array.
independent_snapshots(us) = allunique(objectid(u[1]) for u in us)

@testset "Removed alias_jump keyword" begin
    jprob = bd_jprob()
    ojprob = ode_jprob()
    dae(out, du, u, p, t) = (out .= du; nothing)
    daejprob = JumpProblem(DAEProblem(dae, [0.0], [10.0], (0.0, 1.0)), Direct(),
        COUPLED_JUMP)
    for value in (true, false)
        calls = (() -> solve(jprob, SSAStepper(); alias_jump = value),
            () -> init(jprob, SSAStepper(); alias_jump = value),
            () -> solve(ojprob, Tsit5(); alias_jump = value),
            () -> init(ojprob, Tsit5(); alias_jump = value),
            () -> init(daejprob, DFBDF(); alias_jump = value),
            () -> bd_jprob(; alias_jump = value))
        for call in calls
            @test_throws ArgumentError call()
            @test_throws REMOVED_MSG call()
        end
    end

    # Any supplied value is rejected, including `nothing`.
    @test_throws REMOVED_MSG init(jprob, SSAStepper(); alias_jump = nothing)
    @test_throws REMOVED_MSG solve(ojprob, Tsit5(); alias_jump = nothing)
    @test_throws REMOVED_MSG bd_jprob(; alias_jump = nothing)

    # The keyword is also rejected when stored on the wrapped problem.
    stored_dprob = DiscreteProblem([10], (0.0, 1.0), [1.0, 0.1]; alias_jump = false)
    stored_jprob = JumpProblem(stored_dprob, Direct(), bd_maj())
    @test_throws REMOVED_MSG init(stored_jprob, SSAStepper())
    @test_throws REMOVED_MSG solve(stored_jprob, SSAStepper())
    stored_oprob = ODEProblem((du, u, p, t) -> (du .= 0; nothing), [10.0], (0.0, 1.0);
        alias_jump = false)
    stored_ojprob = JumpProblem(stored_oprob, Direct(), COUPLED_JUMP)
    @test_throws REMOVED_MSG init(stored_ojprob, Tsit5())
    @test_throws REMOVED_MSG solve(stored_ojprob, Tsit5())

    # The plural spelling was never a solve keyword; SciMLBase's keyword validation
    # rejects it before JumpProcesses code runs.
    @test_throws SciMLBase.CommonKwargError solve(jprob, SSAStepper(); alias_jumps = false)
    @test_throws SciMLBase.CommonKwargError init(jprob, SSAStepper(); alias_jumps = false)
    @test_throws SciMLBase.CommonKwargError solve(ojprob, Tsit5(); alias_jumps = false)
    @test_throws SciMLBase.CommonKwargError init(ojprob, Tsit5(); alias_jumps = false)
end

@testset "CPU tau-leaping solvers reject alias keywords" begin
    # These solvers build fresh working state, so neither keyword applies; their strict
    # keyword signatures reject both rather than silently ignoring them.
    leaping = JumpProblem(DiscreteProblem([10], (0.0, 1.0), [1.0, 0.1]), PureLeaping(),
        bd_maj())
    for alg in (SimpleExplicitTauLeaping(), SimpleImplicitTauLeaping(),
        SimpleTrapezoidalLeaping(), SimpleAdaptiveTauLeaping())
        @test_throws MethodError solve(leaping, alg; alias_jump = true)
        @test_throws MethodError solve(leaping, alg; alias = true)
        @test SciMLBase.successful_retcode(solve(leaping, alg; seed = 1))
    end
end

@testset "JumpProcesses-owned solvers reuse the problem's jump state" begin
    jprob = bd_jprob()
    @test uses_problem_callback(init(jprob, SSAStepper()), jprob)
    @test uses_problem_callback(fetch(Threads.@spawn init(jprob, SSAStepper())), jprob)
    @test uses_problem_callback(init(jprob, SSAStepper(); alias = false), jprob)

    ojprob = ode_jprob()
    @test uses_problem_aggregation(init(ojprob, Tsit5()), ojprob)
    @test uses_problem_aggregation(fetch(Threads.@spawn init(ojprob, Tsit5())), ojprob)

    # `alias` is forwarded unchanged to the coupled solver.
    integrator = init(ojprob, Tsit5(); alias = ODEAliasSpecifier(alias_u0 = true))
    @test integrator.u === ojprob.prob.u0
    @test uses_problem_aggregation(integrator, ojprob)
end

@testset "Reuse re-initializes jump state from the current state and parameters" begin
    jprob = bd_jprob()
    aggregation = jprob.discrete_jump_aggregation
    sol = solve(jprob, SSAStepper(); seed = 1)
    @test SciMLBase.successful_retcode(sol)
    @test length(sol.t) > 2

    # Default `alias_p` reuses the problem's parameters, so in-place parameter changes
    # are seen by the next initialization of the same jump state.
    integrator = init(jprob, SSAStepper(); seed = 2)
    @test integrator.u == [10]
    @test aggregation.sum_rate ≈ 1.0 + 0.1 * 10
    jprob.prob.p[1] = 3.0
    integrator = init(jprob, SSAStepper(); seed = 2)
    @test aggregation.maj_rates == [3.0, 0.1]
    @test aggregation.sum_rate ≈ 3.0 + 0.1 * 10
end

@testset "remake shares jump state with the original problem" begin
    jprob = bd_jprob()
    remade = remake(jprob; p = [2.0, 0.2])
    @test remade.discrete_jump_aggregation === jprob.discrete_jump_aggregation
    @test remade.jump_callback === jprob.jump_callback

    # Serial use stays correct: every init re-initializes the shared state from the
    # problem being solved.
    aggregation = jprob.discrete_jump_aggregation
    init(remade, SSAStepper())
    @test aggregation.sum_rate ≈ 2.0 + 0.2 * 10
    init(jprob, SSAStepper())
    @test aggregation.sum_rate ≈ 1.0 + 0.1 * 10
end

@testset "SSAStepper alias controls for u0 and p" begin
    specs = Any[nothing, true, false]
    for alias_u0 in (nothing, true, false), alias_p in (nothing, true, false)

        push!(specs, ODEAliasSpecifier(; alias_u0, alias_p))
        push!(specs, SciMLBase.DiscreteAliasSpecifier(; alias_u0, alias_p))
    end
    choice(spec::Union{Nothing, Bool}, field) = spec
    choice(spec, field) = getfield(spec, field)

    for spec in specs
        jprob = bd_jprob()
        u0, p = jprob.prob.u0, jprob.prob.p
        integrator = init(jprob, SSAStepper(); alias = spec, seed = 3)
        u0_aliased = choice(spec, :alias_u0) === true
        p_aliased = choice(spec, :alias_p) !== false
        @test (integrator.u === u0) == u0_aliased
        @test (integrator.p === p) == p_aliased
        @test integrator.p == p
        # `alias` never controls jump state.
        @test uses_problem_callback(integrator, jprob)

        solve!(integrator)
        @test integrator.sol.u[1] == [10]
        @test u0 == (u0_aliased ? integrator.u : [10])
    end

    @test_throws ArgumentError init(bd_jprob(), SSAStepper();
        alias = SciMLBase.SDEAliasSpecifier())
end

@testset "Parameter-changing callbacks use the selected parameters" begin
    for alias_p in (nothing, true, false)
        jprob = bd_jprob()
        p = jprob.prob.p
        stop_jumps = DiscreteCallback((u, t, integrator) -> t == 1.0,
            function (integrator)
                integrator.p[1] = 0.0
                integrator.p[2] = 0.0
                reset_aggregated_jumps!(integrator)
            end)
        sol = solve(jprob, SSAStepper(); alias = ODEAliasSpecifier(; alias_p),
            callback = stop_jumps, tstops = [1.0], seed = 4)
        @test SciMLBase.successful_retcode(sol)
        @test all(==(sol(1.0)), (sol(t) for t in 1.0:0.5:10.0))
        @test p == (alias_p === false ? [1.0, 0.1] : [0.0, 0.0])
    end
end

@testset "SSAStepper tstops containers and aliasing" begin
    for tstops in ([2.0, 5.0], (2.0, 5.0), 2.0:3.0:5.0, [2, 5], view([0.0, 2.0, 5.0], 2:3))
        integrator = init(bd_jprob(), SSAStepper(); tstops)
        @test integrator.tstops isa Vector{Float64}
        add_tstop!(integrator, 3.0)
        @test integrator.tstops == [2.0, 3.0, 5.0]
        solve!(integrator)
        @test SciMLBase.successful_retcode(integrator.sol)
    end
    integrator = init(bd_jprob(), SSAStepper(); tstops = 4.0)
    add_tstop!(integrator, 6.0)
    @test integrator.tstops == [4.0, 6.0]

    for alias in (nothing, true, false, ODEAliasSpecifier(alias_tstops = true),
        ODEAliasSpecifier(alias_tstops = false), SciMLBase.DiscreteAliasSpecifier())
        user = [2.0, 5.0]
        integrator = init(bd_jprob(), SSAStepper(); tstops = user, alias)
        aliased = (alias === true) ||
                  (alias isa ODEAliasSpecifier && alias.alias_tstops === true)
        copied_at_init = (alias === false) ||
                         (alias isa ODEAliasSpecifier && alias.alias_tstops === false)
        add_tstop!(integrator, 3.0)
        @test integrator.tstops == [2.0, 3.0, 5.0]
        @test user == (aliased ? [2.0, 3.0, 5.0] : [2.0, 5.0])
        @test (integrator.tstops === user) == aliased
        @test !copied_at_init || integrator.tstops !== user
    end

    # Without aliasing permission the caller's array is never mutated, and an
    # explicit `false` copies at `init` so later caller changes are not seen.
    user = [2.0, 5.0]
    integrator = init(bd_jprob(), SSAStepper(); tstops = user,
        alias = ODEAliasSpecifier(alias_tstops = false))
    user[2] = 7.0
    @test integrator.tstops == [2.0, 5.0]

    # Callable tstops are evaluated at init and unaffected by alias controls.
    integrator = init(bd_jprob(), SSAStepper(); tstops = (p, tspan) -> [3.0],
        alias = true)
    @test integrator.tstops == [3.0]
end

@testset "Saved states are independent recursive snapshots" begin
    # A nested state type, where a shallow `copy` would share inner arrays.
    jump = ConstantRateJump((u, p, t) -> 1.0,
        integrator -> (integrator.u[1][1] += 1; nothing))
    nested_prob(save_positions) = JumpProblem(
        DiscreteProblem([[0], [0]], (0.0, 10.0)), Direct(), jump; save_positions)

    for alias_u0 in (nothing, true, false)
        jprob = nested_prob((false, true))
        sol = solve(jprob, SSAStepper(); saveat = 2.0, seed = 5,
            alias = ODEAliasSpecifier(; alias_u0))
        counts = [u[1][1] for u in sol.u]
        @test counts[1] == 0
        @test issorted(counts)
        @test length(unique(counts)) > 1
        @test independent_snapshots(sol.u)

        # Every-step saves and `save_end`.
        jprob = nested_prob((false, true))
        sol = solve(jprob, SSAStepper(); seed = 6, alias = ODEAliasSpecifier(; alias_u0))
        counts = [u[1][1] for u in sol.u]
        # One save per jump, plus a final `save_end` save if the last jump came earlier.
        jump_counts = counts[end] == counts[end - 1] ? counts[1:(end - 1)] : counts
        @test jump_counts == 0:(length(jump_counts) - 1)
        @test independent_snapshots(sol.u)
    end

    # Forced saves from a callback and the integrator's interpolation call.
    jprob = nested_prob((false, false))
    forced = DiscreteCallback((u, t, integrator) -> t == 5.0,
        integrator -> savevalues!(integrator, true))
    integrator = init(jprob, SSAStepper(); callback = forced, tstops = [5.0], seed = 7)
    solve!(integrator)
    @test 5.0 in integrator.sol.t
    @test independent_snapshots(integrator.sol.u)
    snapshot = integrator(integrator.t)
    @test snapshot == integrator.u
    @test snapshot[1] !== integrator.u[1]
    out = [[-1], [-1]]
    integrator(out, integrator.t)
    @test out == integrator.u
    @test out[1] !== integrator.u[1]
end

@testset "In-place state evaluation returns the destination" begin
    # Scalar states are broadcast into the destination.
    scalar_jump = ConstantRateJump((u, p, t) -> 1.0,
        integrator -> (integrator.u += 1; nothing))
    scalar_jprob = JumpProblem(DiscreteProblem(10, (0.0, 1.0)), Direct(), scalar_jump)
    integrator = init(scalar_jprob, SSAStepper(); seed = 1)
    out = [0]
    @test integrator(out, integrator.t) === out
    @test out == [10]
    @test integrator(integrator.t) == 10

    # Flat states are broadcast, converting the element type as before.
    integrator = init(bd_jprob(), SSAStepper(); seed = 1)
    out = zeros(1)
    @test integrator(out, integrator.t) === out
    @test out == [10.0]

    # Arrays of static arrays are copied element-wise.
    no_jump = ConstantRateJump((u, p, t) -> 0.0, integrator -> nothing)
    static_jprob = JumpProblem(DiscreteProblem([SVector(1, 2)], (0.0, 1.0)), Direct(),
        no_jump)
    integrator = init(static_jprob, SSAStepper(); seed = 1)
    out = [SVector(0, 0)]
    @test integrator(out, integrator.t) === out
    @test out == [SVector(1, 2)]
end

# Record which jump aggregator each trajectory uses. The affect and the record live at
# module scope, so copying a problem does not copy them.
const USED_CALLBACKS = Set{UInt}()
const USED_CALLBACKS_LOCK = ReentrantLock()
function recording_affect!(integrator)
    integrator.u[1] += 1
    lock(() -> push!(USED_CALLBACKS, objectid(integrator.cb.condition)), USED_CALLBACKS_LOCK)
    nothing
end

function recording_jprob()
    jump = ConstantRateJump((u, p, t) -> 10.0, recording_affect!)
    JumpProblem(DiscreteProblem([0], (0.0, 1.0)), Direct(), jump)
end

function used_callbacks(f)
    lock(() -> empty!(USED_CALLBACKS), USED_CALLBACKS_LOCK)
    sol = f()
    @test all(SciMLBase.successful_retcode, sol.u)
    lock(() -> copy(USED_CALLBACKS), USED_CALLBACKS_LOCK)
end

@testset "Ensemble copies of jump state" begin
    trajectories = 64
    remake_func(prob, ctx) = remake(prob; u0 = [ctx.sim_id])
    for prob_func in (nothing, remake_func)
        jprob = recording_jprob()
        original = objectid(jprob.discrete_jump_aggregation)
        ensemble(safetycopy) = prob_func === nothing ?
                               EnsembleProblem(jprob; safetycopy) :
                               EnsembleProblem(jprob; prob_func, safetycopy)

        # EnsembleSerial without safety copies reuses the original problem.
        used = used_callbacks(() -> solve(ensemble(false), SSAStepper(),
            EnsembleSerial(); trajectories, seed = 1))
        @test used == Set([original])

        # Safety copies give every trajectory its own copy, made before `prob_func`.
        used = used_callbacks(() -> solve(ensemble(true), SSAStepper(),
            EnsembleSerial(); trajectories, seed = 1))
        @test length(used) == trajectories
        @test original ∉ used

        # EnsembleThreads copies once per spawned task, and JumpProcesses adds no
        # copies; with one thread it falls back to the serial path.
        used = used_callbacks(() -> solve(ensemble(false), SSAStepper(),
            EnsembleThreads(); trajectories, seed = 1))
        if Threads.nthreads() == 1
            @test used == Set([original])
        else
            @test 1 <= length(used) <= Threads.nthreads()
            @test original ∉ used
        end

        used = used_callbacks(() -> solve(ensemble(true), SSAStepper(),
            EnsembleThreads(); trajectories, seed = 1))
        @test length(used) == trajectories
        @test original ∉ used
    end
end

@testset "SortingDirect ensembles replay from identical learned state" begin
    paths(sol) = [(trajectory.t, trajectory.u) for trajectory in sol.u]
    template = bd_jprob(; aggregator = SortingDirect(), u0 = [50], p = [5.0, 0.1])
    for ensemblealg in (EnsembleSerial(), EnsembleThreads())
        # Each replay starts from an independent copy of the same never-solved problem,
        # with the same seed, batching, and thread count.
        run(seed) = solve(EnsembleProblem(deepcopy(template)), SSAStepper(),
            ensemblealg; trajectories = 32, batch_size = 16, seed)
        @test paths(run(11)) == paths(run(11))
        @test paths(run(11)) != paths(run(12))
    end
end
