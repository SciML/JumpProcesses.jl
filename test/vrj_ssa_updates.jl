using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, SciMLBase, StableRNGs, Statistics,
      Test

# Births by a VariableRateJump at rate K(1 + sin Ωt), deaths by a mass action jump at rate ΓA,
# from A = 0: A(t) is Poisson with mean m(t). With a VariableRateJump map the aggregator
# updates the death rate after each birth; without one it keeps the stale rate.
const K, Ω, Γ = 5.0, 2.0, 0.5
function m_exact(t)
    K / Γ * (1 - exp(-Γ * t)) +
    K * (Γ * sin(Ω * t) - Ω * cos(Ω * t) + Ω * exp(-Γ * t)) / (Γ^2 + Ω^2)
end
birth_rate(u, p, t) = K * (1 + sin(Ω * t))
birth!(integ) = (integ.u[1] += 1; nothing)
decay = MassActionJump([Γ], [[1 => 1]], [[1 => -1]])
zero_drift!(du, u, p, t) = (fill!(du, 0); nothing)
noise!(du, u, p, t) = (fill!(du, 0); du[2] = 1; nothing)    # on a component no rate reads
const T = 5.0
const VR_AGGREGATORS = (VR_FRM(), VR_Direct(), VR_DirectFW())

function birth_death(agg, vr_aggregator; sde = false, deps = nothing)
    deps = deps === nothing ? JumpDependencies(; vrjtovars_map = [[1]]) : deps
    vrj = VariableRateJump(birth_rate, birth!; save_positions = (false, false))
    prob = sde ? SDEProblem(zero_drift!, noise!, [0.0, 0.0], (0.0, T)) :
           ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [0.0], (0.0, T))
    JumpProblem(prob, agg, decay, vrj; vr_aggregator, dependencies = deps,
        save_positions = (false, false))
end

function final_A(jprob, alg, n; seed = 0)
    [solve(jprob, alg; rng = StableRNG(seed + i),
         save_everystep = false).u[end][1] for i in 1:n]
end

# the mean of n samples of A(T) is within 4 standard errors of m(T), and A(T) is Poissonian
function poisson_with_exact_mean(A)
    m = m_exact(T)
    abs(mean(A) - m) < 4 * sqrt(m / length(A)) && 0.8 < var(A) / mean(A) < 1.2
end

@testset "SSA jumps are updated after a VariableRateJump occurs (ODE)" begin
    for vr_aggregator in VR_AGGREGATORS
        for agg in JumpProcesses.JUMP_AGGREGATORS
            A = final_A(birth_death(agg, vr_aggregator), Tsit5(), 600)
            @test poisson_with_exact_mean(A)
        end
    end
end

@testset "SSA jumps are updated after a VariableRateJump occurs (SDE)" begin
    for vr_aggregator in (VR_FRM(), VR_Direct())
        for agg in (Direct(), NRM(), RSSA(), Coevolve())
            A = final_A(birth_death(agg, vr_aggregator; sde = true), SRIW1(), 400)
            @test poisson_with_exact_mean(A)
        end
    end
end

@testset "Without VariableRateJump maps the SSA jumps aren't updated, as before" begin
    for agg in (Direct(), NRM(), RSSA())
        deps = JumpDependencies(; dep_graph = [[1]])
        A = final_A(birth_death(agg, VR_FRM(); deps), Tsit5(), 200)
        # no death ever: the death rate stays at its value at A = 0
        @test mean(A) > m_exact(T) + 10
    end
end

# A VariableRateJump whose entry lists nothing changes no rate an SSA jump reads, so the SSA
# aggregator is left alone: the same seeded results as without the maps
@testset "Empty entries need no update" begin
    count!(integ) = (integ.u[2] += 1; nothing)
    for vr_aggregator in VR_AGGREGATORS
        for agg in (Direct(), NRM(), RSSA(), SortingDirect())
            function prob(deps)
                vrj = VariableRateJump((u, p, t) -> 1 + sin(t), count!;
                    save_positions = (false, false))
                oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!,
                    [10.0, 0.0], (0.0, T))
                JumpProblem(oprob, agg, decay, vrj; vr_aggregator, dependencies = deps,
                    save_positions = (false, false))
            end
            v2j = [[1], Int[]]
            with = agg isa RSSA ?
                   JumpDependencies(; vrjtovars_map = [[2]], vartojumps_map = v2j,
                jumptovars_map = [[1]]) :
                   JumpDependencies(; vrj_dep_graph = [Int[]], vrjtovars_map = [[2]],
                vartojumps_map = v2j, jumptovars_map = [[1]], dep_graph = [[1]])
            without = JumpDependencies(; vartojumps_map = v2j, jumptovars_map = [[1]],
                dep_graph = [[1]])
            sol1 = solve(prob(with), Tsit5(); rng = StableRNG(1), save_everystep = false)
            sol2 = solve(prob(without), Tsit5(); rng = StableRNG(1), save_everystep = false)
            @test sol1.u[end] == sol2.u[end]
        end
    end
end

# Each VariableRateJump creates an A, and A → B at rate 1e20, so B's time rounds to the
# VariableRateJump's: the update must let it fire at once, with no stall
@testset "An SSA jump due at the VariableRateJump's time fires at once" begin
    create!(integ) = (integ.u[1] += 1; integ.u[3] += 1; nothing)
    conv = MassActionJump([1e20], [[1 => 1]], [[1 => -1, 2 => 1]])
    tie_noise!(du, u, p, t) = (fill!(du, 0); du[4] = 1; nothing)
    for sde in (false, true)
        for vr_aggregator in (VR_FRM(), VR_Direct())
            for agg in (Direct(), NRM(), RSSA(), Coevolve())
                vrj = VariableRateJump((u, p, t) -> 1.0, create!;
                    save_positions = (false, false))
                prob = sde ? SDEProblem(zero_drift!, tie_noise!, zeros(4), (0.0, 10.0)) :
                       ODEProblem(zero_drift!, zeros(3), (0.0, 10.0))
                deps = JumpDependencies(; vrjtovars_map = [[1, 3]])
                jprob = JumpProblem(prob, agg, conv, vrj; vr_aggregator,
                    dependencies = deps, save_positions = (false, false))
                alg = sde ? SRIW1() : Tsit5()
                ends = [solve(jprob, alg; rng = StableRNG(i), save_everystep = false).u[end]
                        for i in 1:200]
                @test all(u -> u[1] == 0 && u[2] == u[3], ends)
            end
        end
    end
end

# G is switched on and off by two VariableRateJumps, and P is produced at rate 10 G by a mass
# action jump: P must only be produced while G is on, so the production rate has to go from
# zero to positive and back after the VariableRateJumps
@testset "An SSA rate switched on and off by VariableRateJumps" begin
    on!(integ) = (integ.u[1] = 1; nothing)
    off!(integ) = (integ.u[1] = 0; nothing)
    production = MassActionJump([10.0], [[1 => 1]], [[2 => 1]])
    for vr_aggregator in VR_AGGREGATORS
        for agg in JumpProcesses.JUMP_AGGREGATORS
            vrjs = (VariableRateJump((u, p, t) -> 1 - u[1], on!),
                VariableRateJump((u, p, t) -> u[1], off!))
            oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [0.0, 0.0],
                (0.0, 20.0))
            deps = JumpDependencies(; vrjtovars_map = [[1], [1]])
            jprob = JumpProblem(oprob, agg, production, vrjs...; vr_aggregator,
                dependencies = deps, save_positions = (false, true))
            sol = solve(jprob, Tsit5(); rng = StableRNG(1), save_everystep = false)
            produced = [i for i in 2:length(sol.u) if sol.u[i][2] > sol.u[i - 1][2]]
            produced_while = [sol.u[i][1] for i in produced]
            @test !isempty(produced_while)
            @test all(==(1), produced_while)
        end
    end
end

# With Coevolve the bounded VariableRateJump (C, at rate 1) is an SSA jump, so its entries are
# ignored and the VR aggregator handles only the general one (the births of A)
@testset "Bounded VariableRateJumps handled by Coevolve" begin
    bump!(integ) = (integ.u[2] += 1; nothing)
    for vr_aggregator in VR_AGGREGATORS
        bounded = VariableRateJump((u, p, t) -> 1.0, bump!; urate = (u, p, t) -> 1.0,
            rateinterval = (u, p, t) -> Inf)
        general = VariableRateJump(birth_rate, birth!; save_positions = (false, false))
        oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [0.0, 0.0],
            (0.0, T))
        deps = JumpDependencies(; dep_graph = [[1], [2]], vartojumps_map = [[1], Int[]],
            vrjtovars_map = [[2], [1]])
        jprob = JumpProblem(oprob, Coevolve(), decay, bounded, general; vr_aggregator,
            dependencies = deps, save_positions = (false, false))
        ends = [solve(jprob, Tsit5(); rng = StableRNG(i), save_everystep = false).u[end]
                for i in 1:600]
        @test poisson_with_exact_mean([u[1] for u in ends])
        @test mean(u[2] for u in ends)≈T atol=0.5
    end
end

@testset "Missing or wrong VariableRateJump maps" begin
    vrj = VariableRateJump(birth_rate, birth!)
    crj = ConstantRateJump((u, p, t) -> Γ * u[1], integ -> (integ.u[1] -= 1; nothing))
    oprob = ODEProblem(zero_drift!, [0.0], (0.0, T))
    # a dependency-list aggregator with a constant rate jump can't derive vrj_dep_graph
    # without vartojumps_map
    deps = JumpDependencies(; vrjtovars_map = [[1]], dep_graph = [[1]])
    msg = "The NRM algorithm needs the jumps that depend on each `VariableRateJump` to " *
          "update them after it occurs: give `vrj_dep_graph`, or `vrjtovars_map` and " *
          "`vartojumps_map`, in the `JumpDependencies`."
    @test_throws ErrorException(msg) JumpProblem(oprob, NRM(), crj, vrj;
        dependencies = deps)
    deps = JumpDependencies(; vrjtovars_map = [[1]], vartojumps_map = [[1]],
        dep_graph = [[1]])
    @test JumpProblem(oprob, NRM(), crj, vrj; dependencies = deps) isa JumpProblem
    # RSSA needs the changed variables
    deps = JumpDependencies(; vrj_dep_graph = [[1]])
    msg = "The RSSA algorithm needs the variables each `VariableRateJump` changes to " *
          "update its jumps after one occurs: give `vrjtovars_map` in the " *
          "`JumpDependencies`."
    @test_throws ErrorException(msg) JumpProblem(oprob, RSSA(), decay, vrj;
        dependencies = deps)
    # Direct needs nothing more
    deps = JumpDependencies(; vrjtovars_map = [[1]])
    @test JumpProblem(oprob, Direct(), crj, vrj; dependencies = deps) isa JumpProblem
    # one entry per VariableRateJump, with valid indices
    bad = (JumpDependencies(; vrjtovars_map = [[1], [1]]),
        JumpDependencies(; vrjtovars_map = [[2]]),
        JumpDependencies(; vrj_dep_graph = [[2]]))
    for deps in bad
        @test_throws ArgumentError JumpProblem(oprob, Direct(), decay, vrj;
            dependencies = deps)
    end
end

# The VR callbacks find the SSA aggregator among the integrator's discrete callbacks each
# time a VariableRateJump occurs; over a Tuple the search must resolve at compile time,
# however many callbacks there are. A search through `Base.tail` stopped doing so beyond 32
# callbacks: still inferred and allocation free, it took about 10 µs among 41 callbacks
# instead of a few ns. So the best of 5 runs of 10^4 searches must take under 1 ms (100 ns
# per search), with a wide margin on both sides, coverage runs included. Each result is
# stored, so that no search can be skipped.
function search_loop(cbs, n, sink)
    for _ in 1:n
        sink[] = JumpProcesses.find_ssa_aggregator(cbs)
    end
    nothing
end
function search_time(cbs; n = 10^4)
    sink = Ref(JumpProcesses.find_ssa_aggregator(cbs))
    search_loop(cbs, n, sink)
    minimum(@elapsed(search_loop(cbs, n, sink)) for _ in 1:5)
end

@testset "Finding the SSA aggregator among many callbacks" begin
    jprob = birth_death(NRM(), VR_FRM())
    never = [DiscreteCallback((u, t, integ) -> false, integ -> nothing) for _ in 1:40]
    cbs = CallbackSet(never...)
    integ = init(jprob, Tsit5(); rng = StableRNG(1), callback = cbs)
    dcbs = integ.opts.callback.discrete_callbacks
    @test length(dcbs) == 41
    @test (@inferred JumpProcesses.find_ssa_aggregator(dcbs)) ===
          jprob.discrete_jump_aggregation
    @test search_time(dcbs) < 1e-3
    # with the user's callbacks first, as StochasticDiffEq orders them
    dcbs2 = (cbs.discrete_callbacks..., dcbs[1])
    @test (@inferred JumpProcesses.find_ssa_aggregator(dcbs2)) ===
          jprob.discrete_jump_aggregation
    @test search_time(dcbs2) < 1e-3
    JumpProcesses.find_ssa_aggregator(dcbs2)
    @test (@allocated JumpProcesses.find_ssa_aggregator(dcbs2)) == 0
    @test JumpProcesses.find_ssa_aggregator(cbs.discrete_callbacks) === nothing
    # the whole problem solves with them
    A = [solve(jprob, Tsit5(); rng = StableRNG(i), callback = cbs,
             save_everystep = false).u[end][1] for i in 1:400]
    @test poisson_with_exact_mean(A)
end

@testset "remake and ensembles keep the updates" begin
    jprob = birth_death(NRM(), VR_Direct())
    sol1 = solve(remake(jprob; u0 = [3.0]), Tsit5(); rng = StableRNG(7),
        save_everystep = false)
    oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [3.0], (0.0, T))
    vrj = VariableRateJump(birth_rate, birth!; save_positions = (false, false))
    dependencies = JumpDependencies(; vrjtovars_map = [[1]])
    jprob2 = JumpProblem(oprob, NRM(), decay, vrj; vr_aggregator = VR_Direct(),
        dependencies, save_positions = (false, false))
    sol2 = solve(jprob2, Tsit5(); rng = StableRNG(7), save_everystep = false)
    @test sol1.u[end] == sol2.u[end]
    eprob = EnsembleProblem(jprob)
    esol = solve(eprob, Tsit5(), EnsembleSerial(); trajectories = 400,
        rng = StableRNG(3), save_everystep = false)
    @test poisson_with_exact_mean([s.u[end][1] for s in esol.u])
end

# CCNRM keeps its jump times in bins and searches from the bin where its last search stopped.
# A VariableRateJump occurs before the next SSA jump, possibly in an earlier bin, so the jump
# it enables must still be found. Here X, made by the VariableRateJump, becomes Y at once (at
# rate 1e8), and a slow independent reaction C → ∅ is the next SSA jump: in a later bin of the
# first window (rate 1), or after the first window, so that the table's window is rebuilt to
# start at its time (rate 0.05). A fast F → ∅ sets a short bin width at the start. Narrow
# bins and few of them (`binwidthconst = 2, numbinsconst = 5`) also make the update rebuild
# the window to start at the VariableRateJump's time, from bins that are then all empty.
@testset "CCNRM finds a jump that a VariableRateJump moves before its search position" begin
    bins = ((;), (; binwidthconst = 2, numbinsconst = 5))
    for crate in (1.0, 0.05), vr_aggregator in (VR_FRM(), VR_Direct()), kw in bins
        maj = MassActionJump([1e8, crate, 100.0], [[1 => 1], [3 => 1], [4 => 1]],
            [[1 => -1, 2 => 1], [3 => -1], [4 => -1]])
        vrj = VariableRateJump((u, p, t) -> 2.0, birth!; save_positions = (false, false))
        oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [0.0, 0.0, 1.0, 1.0],
            (0.0, 2.0))
        jprob = JumpProblem(oprob, CCNRM(), maj, vrj; vr_aggregator,
            dependencies = JumpDependencies(; vrjtovars_map = [[1]]),
            save_positions = (false, false), kw...)
        sols = [solve(jprob, Tsit5(); rng = StableRNG(i), save_everystep = false)
                for i in 1:100]
        @test all(sol -> SciMLBase.successful_retcode(sol), sols)
        # every X made has become a Y
        @test all(sol -> sol.u[end][1] == 0, sols)
        @test sum(sol -> sol.u[end][2], sols) > 0
    end
end

# After a VariableRateJump's affect stops the solve, the update must not schedule the SSA jump
# it enables (X → Y at rate 1e6), which would restart the solve.
@testset "No update after a VariableRateJump stops the solve" begin
    stop!(integ) = (integ.u[1] += 1; terminate!(integ); nothing)
    stop_custom!(integ) = (integ.u[1] += 1; terminate!(integ, ReturnCode.Success); nothing)
    maj = MassActionJump([1e6], [[1 => 1]], [[1 => -1, 2 => 1]])
    deps = JumpDependencies(; vrjtovars_map = [[1]])
    oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [0.0, 0.0], (0.0, 10.0))
    sprob = SDEProblem{true, SciMLBase.FullSpecialize}(zero_drift!,
        (du, u, p, t) -> (fill!(du, 0); nothing), [0.0, 0.0], (0.0, 10.0))
    cases = ((oprob, Tsit5(), stop!, ReturnCode.Terminated, (;)),
        (oprob, Tsit5(), stop_custom!, ReturnCode.Success, (;)),
        (sprob, EM(), stop!, ReturnCode.Terminated, (; dt = 0.01)))
    for (prob, alg, affect!, retcode, kw) in cases, vr_aggregator in VR_AGGREGATORS
        vrj = VariableRateJump((u, p, t) -> 5.0, affect!)
        jprob = JumpProblem(prob, Direct(), maj, vrj; vr_aggregator, dependencies = deps)
        sol = solve(jprob, alg; rng = StableRNG(1), kw...)
        @test sol.retcode == retcode
        # the solve stopped right after the VariableRateJump: X was made and not converted
        @test sol.u[end][1:2] == [1.0, 0.0]
        @test sol.t[end] < 10.0
    end
end

# `isolated_copy` shares the update lists, which are read-only, and copies the working state.
@testset "isolated_copy keeps the updates" begin
    for vr_aggregator in VR_AGGREGATORS, agg in (Direct(), NRM())
        jprob = birth_death(agg, vr_aggregator)
        jcopy = isolated_copy(jprob)
        sol1 = solve(jprob, Tsit5(); rng = StableRNG(5), save_everystep = false)
        sol2 = solve(jcopy, Tsit5(); rng = StableRNG(5), save_everystep = false)
        @test sol1.u[end] == sol2.u[end]
        if !(vr_aggregator isa VR_FRM)
            cache = jprob.jump_callback.continuous_callbacks[1].condition
            cache2 = jcopy.jump_callback.continuous_callbacks[1].condition
            @test cache2 !== cache
            @test cache2.updates === cache.updates
            @test cache2.cum_rate_sum !== cache.cum_rate_sum
        end
    end
    @test poisson_with_exact_mean(final_A(isolated_copy(birth_death(NRM(), VR_Direct())),
        Tsit5(), 400; seed = 2000))
end

# An update that leaves the next jump time unchanged keeps its stop instead of adding another:
# with NRM, a VariableRateJump that changes the rate of a jump other than the next one.
@testset "An unchanged next jump time adds no stop" begin
    # 1: X → ∅ (rate 1e-6 X, far in the future), 2: Z → ∅ (rate 1, the next jump)
    maj = MassActionJump([1e-6, 1.0], [[1 => 1], [2 => 1]], [[1 => -1], [2 => -1]])
    vrj = VariableRateJump((u, p, t) -> 1.0, birth!)
    oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [1.0, 1.0], (0.0, 1e3))
    jprob = JumpProblem(oprob, NRM(), maj, vrj; vr_aggregator = VR_Direct(),
        dependencies = JumpDependencies(; vrjtovars_map = [[1]]))
    integ = init(jprob, Tsit5(); rng = StableRNG(1))
    agg = jprob.discrete_jump_aggregation
    ntstops = length(integ.opts.tstops)
    next_time = agg.next_jump_time
    for _ in 1:100
        JumpProcesses.update_after_external_change!(agg, integ, [1])
    end
    @test agg.next_jump == 2 && agg.next_jump_time == next_time
    @test length(integ.opts.tstops) == ntstops
end

# The aggregators that recompute every rate only store whether each VariableRateJump changes
# a rate; the others store the lists.
@testset "What the VR callbacks store" begin
    deps = JumpDependencies(; vrj_dep_graph = [[1], Int[]])
    vrjs = (VariableRateJump(birth_rate, birth!), VariableRateJump(birth_rate, birth!))
    oprob = ODEProblem{true, SciMLBase.FullSpecialize}(zero_drift!, [0.0], (0.0, T))
    for agg in (Direct(), DirectFW(), FRM(), FRMFW())
        jprob = JumpProblem(oprob, agg, decay, vrjs...; vr_aggregator = VR_Direct(),
            dependencies = deps)
        @test jprob.jump_callback.continuous_callbacks[1].condition.updates.lists ==
              [true, false]
    end
    jprob = JumpProblem(oprob, NRM(), decay, vrjs...; vr_aggregator = VR_Direct(),
        dependencies = deps)
    @test jprob.jump_callback.continuous_callbacks[1].condition.updates.lists == [[1], Int[]]
end
