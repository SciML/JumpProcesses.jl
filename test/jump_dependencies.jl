using JumpProcesses, StableRNGs, Test

# ∅ → A, A → ∅ and B → ∅ as mass action jumps 1-3, and A → B (rate 0.5 A) as constant rate
# jump 4, so the maps can't be generated from the mass action jumps
maj = MassActionJump([10.0, 1.0, 0.2], [Pair{Int, Int}[], [1 => 1], [2 => 1]],
    [[1 => 1], [1 => -1], [2 => -1]])
crj = ConstantRateJump((u, p, t) -> 0.5 * u[1], integ -> (integ.u[1] -= 1;
    integ.u[2] += 1; nothing))
prob = DiscreteProblem([5, 0], (0.0, 20.0))
vartojumps_map() = [[2, 4], [3]]
jumptovars_map() = [[1], [1], [2], [1, 2]]
dep_graph() = [[2, 4], [2, 4], [3], [2, 3, 4]]   # jump 1 is missing from its own entry

@testset "JumpDependencies copies the maps unless aliased" begin
    g = dep_graph()
    deps = JumpDependencies(; dep_graph = g)
    @test deps.dep_graph == g
    @test deps.dep_graph !== g
    @test deps.dep_graph[1] !== g[1]
    @test deps.vartojumps_map === nothing
    @test deps.jumptovars_map === nothing
    @test JumpDependencies(; dep_graph = g, alias = true).dep_graph === g

    # building and solving a problem leaves the user's maps as they were
    jprob = JumpProblem(prob, NRM(), maj, crj; dependencies = deps)
    solve(jprob, SSAStepper(); rng = StableRNG(1))
    @test g == dep_graph()
end

@testset "JumpDependencies and the keywords give identical results" begin
    for agg in JumpProcesses.JUMP_AGGREGATORS
        deps = JumpDependencies(; vartojumps_map = vartojumps_map(),
            jumptovars_map = jumptovars_map(), dep_graph = dep_graph())
        jprob1 = JumpProblem(prob, agg, maj, crj; dependencies = deps)
        jprob2 = JumpProblem(prob, agg, maj, crj; vartojumps_map = vartojumps_map(),
            jumptovars_map = jumptovars_map(), dep_graph = dep_graph())
        sol1 = solve(jprob1, SSAStepper(); rng = StableRNG(1))
        sol2 = solve(jprob2, SSAStepper(); rng = StableRNG(1))
        @test sol1.t == sol2.t
        @test sol1.u == sol2.u
    end
end

@testset "JumpDependencies with Coevolve and bounded VariableRateJumps" begin
    # a fifth jump, A → ∅ at a rate rising with t, bounded on [t, t + 1]
    vrj = VariableRateJump((u, p, t) -> u[1] * t / 20, integ -> (integ.u[1] -= 1; nothing);
        urate = (u, p, t) -> u[1] * (t + 1) / 20, rateinterval = (u, p, t) -> 1.0)
    deps = JumpDependencies(;
        dep_graph = [[2, 4, 5], [2, 4, 5], [3], [2, 3, 4, 5],
        [2, 4, 5]])
    jprob1 = JumpProblem(prob, Coevolve(), maj, crj, vrj; dependencies = deps)
    jprob2 = JumpProblem(prob, Coevolve(), maj, crj, vrj;
        dep_graph = [[2, 4, 5], [2, 4, 5], [3], [2, 3, 4, 5], [2, 4, 5]])
    sol1 = solve(jprob1, SSAStepper(); rng = StableRNG(1))
    sol2 = solve(jprob2, SSAStepper(); rng = StableRNG(1))
    @test sol1.t == sol2.t
    @test sol1.u == sol2.u
    # the bounded VariableRateJump has an entry, so four entries are too few
    deps4 = JumpDependencies(; dep_graph = dep_graph())
    @test_throws ArgumentError JumpProblem(prob, Coevolve(), maj, crj, vrj;
        dependencies = deps4)
end

# with 30 constant rate jumps, the maps decide which aggregator JumpProblem picks
@testset "JumpProblem picks the aggregator from JumpDependencies" begin
    n = 30
    crjs = [ConstantRateJump((u, p, t) -> 1.0, integ -> (integ.u[1] += 1; nothing))
            for _ in 1:n]
    p1 = DiscreteProblem([0], (0.0, 1.0))
    g = [Int[] for _ in 1:n]
    v2j, j2v = [Int[]], [[1] for _ in 1:n]
    jprob = JumpProblem(p1, JumpSet(crjs...); dependencies = JumpDependencies(;
        dep_graph = g))
    @test jprob.aggregator isa SortingDirect
    @test JumpProblem(p1, JumpSet(crjs...); dep_graph = g).aggregator isa SortingDirect
    jprob = JumpProblem(p1, JumpSet(crjs...);
        dependencies = JumpDependencies(;
            vartojumps_map = v2j, jumptovars_map = j2v))
    @test jprob.aggregator isa RSSA
end

@testset "A map given both ways is an error" begin
    deps = JumpDependencies(; dep_graph = dep_graph())
    @test_throws ArgumentError JumpProblem(prob, NRM(), maj, crj; dependencies = deps,
        dep_graph = dep_graph())
    @test_throws ArgumentError JumpProblem(prob, maj, crj; dependencies = deps,
        dep_graph = dep_graph())
    # different maps can come different ways
    deps = JumpDependencies(; vartojumps_map = vartojumps_map())
    jprob = JumpProblem(prob, RSSA(), maj, crj; dependencies = deps,
        jumptovars_map = jumptovars_map())
    @test jprob.discrete_jump_aggregation isa JumpProcesses.RSSAJumpAggregation
end

@testset "JumpProblem checks the maps of a JumpDependencies" begin
    bad_maps = (
        (; dep_graph = [[2, 4], [2, 4], [3]]),               # 3 entries for 4 jumps
        (; dep_graph = [[2, 4], [2, 5], [3], [2, 3, 4]]),     # jump 5 doesn't exist
        (; dep_graph = [[2, 4], [0], [3], [2, 3, 4]]),        # nor jump 0
        (; vartojumps_map = [[2, 4], [3], Int[]]),            # 3 entries for 2 variables
        (; vartojumps_map = [[2, 4], [3, 7]]),                # jump 7 doesn't exist
        (; jumptovars_map = [[1], [1], [2]]),                 # 3 entries for 4 jumps
        (; jumptovars_map = [[1], [1], [3], [1, 2]]))         # variable 3 doesn't exist
    for maps in bad_maps
        deps = JumpDependencies(; maps...)
        @test_throws ArgumentError JumpProblem(
            prob, Direct(), maj, crj; dependencies = deps)
    end
    # the same maps given as keywords are not checked, as before JumpDependencies
    for maps in bad_maps
        @test JumpProblem(prob, Direct(), maj, crj; maps...) isa JumpProblem
    end
end

@testset "remake keeps the dependencies" begin
    deps = JumpDependencies(; dep_graph = dep_graph())
    jprob = JumpProblem(prob, NRM(), maj, crj; dependencies = deps)
    sol1 = solve(remake(jprob; u0 = [20, 3]), SSAStepper(); rng = StableRNG(2))
    jprob2 = JumpProblem(remake(prob; u0 = [20, 3]), NRM(), maj, crj;
        dependencies = JumpDependencies(; dep_graph = dep_graph()))
    sol2 = solve(jprob2, SSAStepper(); rng = StableRNG(2))
    @test sol1.t == sol2.t
    @test sol1.u == sol2.u
end
