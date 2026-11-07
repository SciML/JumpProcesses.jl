using JumpProcesses, StableRNGs, Test

# ∅ → A, A → ∅ and B → ∅ as mass action jumps 1-3, and A → B (rate 0.5 A) as constant rate
# jump 4, so the maps can't be generated from the mass action jumps
maj = MassActionJump([10.0, 1.0, 0.2], [Pair{Int, Int}[], [1 => 1], [2 => 1]],
    [[1 => 1], [1 => -1], [2 => -1]])
crj = ConstantRateJump((u, p, t) -> 0.5 * u[1],
    integ -> (integ.u[1] -= 1; integ.u[2] += 1; nothing))
prob = DiscreteProblem([5, 0], (0.0, 20.0))

# the maps in normal form: entries sorted, without duplicates, each jump in its own entry of
# dep_graph
vartojumps_map() = [[2, 4], [3]]
jumptovars_map() = [[1], [1], [2], [1, 2]]
dep_graph() = [[1, 2, 4], [2, 4], [3], [2, 3, 4]]
# the same maps out of order, with duplicates, and jump 1 missing from its own entry
raw_vartojumps_map() = [[4, 2, 2], [3]]
raw_jumptovars_map() = [[1], [1, 1], [2], [2, 1]]
raw_dep_graph() = [[4, 2], [2, 4], [3, 3], [4, 3, 2]]

function normal_deps()
    JumpDependencies(; vartojumps_map = vartojumps_map(),
        jumptovars_map = jumptovars_map(), dep_graph = dep_graph())
end
function raw_deps()
    JumpDependencies(; vartojumps_map = raw_vartojumps_map(),
        jumptovars_map = raw_jumptovars_map(), dep_graph = raw_dep_graph())
end

function same_solution(jprob1, jprob2; seed = 1)
    sol1 = solve(jprob1, SSAStepper(); rng = StableRNG(seed))
    sol2 = solve(jprob2, SSAStepper(); rng = StableRNG(seed))
    sol1.t == sol2.t && sol1.u == sol2.u
end

@testset "JumpDependencies copies the maps unless aliased" begin
    g = raw_dep_graph()
    deps = JumpDependencies(; dep_graph = g)
    @test deps.dep_graph == g
    @test deps.dep_graph !== g
    @test deps.dep_graph[1] !== g[1]
    @test deps.vartojumps_map === nothing
    @test deps.jumptovars_map === nothing
    @test JumpDependencies(; dep_graph = g, alias = true).dep_graph === g
    deps2 = copy(deps)
    @test deps2.dep_graph == deps.dep_graph
    @test deps2.dep_graph[1] !== deps.dep_graph[1]
end

@testset "A JumpProblem owns the JumpDependencies passed to it" begin
    g = raw_dep_graph()
    deps = JumpDependencies(; dep_graph = g)
    jprob = JumpProblem(prob, NRM(), maj, crj; dependencies = deps)
    # by default the problem puts the maps into normal form in place
    @test deps.dep_graph == dep_graph()
    @test jprob.discrete_jump_aggregation.dep_gr === deps.dep_graph
    # the user's maps were copied when the JumpDependencies was built
    @test g == raw_dep_graph()

    # a map already in normal form is used without a copy
    deps = normal_deps()
    jprob = JumpProblem(prob, RSSA(), maj, crj; dependencies = deps,
        alias_dependencies = false)
    @test jprob.discrete_jump_aggregation.vartojumps_map === deps.vartojumps_map
    @test jprob.discrete_jump_aggregation.jumptovars_map === deps.jumptovars_map
end

@testset "With alias_dependencies = false a JumpDependencies is reused unchanged" begin
    deps = raw_deps()
    for agg in JumpProcesses.JUMP_AGGREGATORS
        jprob1 = JumpProblem(prob, agg, maj, crj; dependencies = deps,
            alias_dependencies = false)
        jprob2 = JumpProblem(prob, agg, maj, crj; dependencies = normal_deps())
        @test same_solution(jprob1, jprob2)
        @test deps.vartojumps_map == raw_vartojumps_map()
        @test deps.jumptovars_map == raw_jumptovars_map()
        @test deps.dep_graph == raw_dep_graph()
    end
end

@testset "Maps out of normal form give the results of normal ones" begin
    for agg in JumpProcesses.JUMP_AGGREGATORS
        jprob1 = JumpProblem(prob, agg, maj, crj; dependencies = raw_deps())
        jprob2 = JumpProblem(prob, agg, maj, crj; dependencies = normal_deps())
        @test same_solution(jprob1, jprob2)
    end
end

@testset "The dependency keywords are deprecated" begin
    @test_deprecated JumpProblem(prob, NRM(), maj, crj; dep_graph = dep_graph())
    # they give the results of a JumpDependencies, and may be put into normal form in place
    for agg in JumpProcesses.JUMP_AGGREGATORS
        v2j, j2v, dg = raw_vartojumps_map(), raw_jumptovars_map(), raw_dep_graph()
        jprob1 = @test_deprecated JumpProblem(prob, agg, maj, crj; vartojumps_map = v2j,
            jumptovars_map = j2v, dep_graph = dg)
        jprob2 = JumpProblem(prob, agg, maj, crj; dependencies = normal_deps())
        @test same_solution(jprob1, jprob2)
        JumpProcesses.needs_depgraph(agg) && @test dg == dep_graph()
        if JumpProcesses.needs_vartojumps_map(agg)
            @test v2j == vartojumps_map()
            @test j2v == jumptovars_map()
        end
    end
end

@testset "Missing maps are generated or are an error" begin
    # only mass action jumps: every aggregator generates the maps it needs
    for agg in JumpProcesses.JUMP_AGGREGATORS
        jprob1 = JumpProblem(prob, agg, maj)
        deps = JumpDependencies(; vartojumps_map = [[2], [3]],
            jumptovars_map = [[1], [1], [2]], dep_graph = [[1, 2], [2], [3]])
        jprob2 = JumpProblem(prob, agg, maj; dependencies = deps)
        @test same_solution(jprob1, jprob2)
    end
    # with a constant rate jump the aggregators that need a map raise their errors
    msg = "To use ConstantRateJumps with the Next Reaction Method (NRM) algorithm a " *
          "dependency graph must be supplied."
    @test_throws ErrorException(msg) JumpProblem(prob, NRM(), maj, crj)
    msg = "To use Coevolve a dependency graph between jumps must be supplied."
    @test_throws ErrorException(msg) JumpProblem(prob, Coevolve(), maj, crj)
    msg = "To use the RSSA algorithm a map from variables to dependent jumps must be supplied."
    @test_throws ErrorException(msg) JumpProblem(prob, RSSA(), maj, crj)
    msg = "To use the RSSACR algorithm a map from jumps to dependent variables must be " *
          "supplied."
    deps = JumpDependencies(; vartojumps_map = vartojumps_map())
    @test_throws ErrorException(msg) JumpProblem(prob, RSSACR(), maj, crj;
        dependencies = deps)
end

@testset "Maps that aren't vectors of vectors" begin
    # entries that are ranges, already in normal form, are used as given
    dg_ranges = [1:4, 2:4, 3:3, 2:4]
    v2j_tuple = ([2, 4], [3])
    j2v_ranges = [1:1, 1:1, 2:2, 1:2]
    for agg in JumpProcesses.JUMP_AGGREGATORS
        deps = JumpDependencies(; vartojumps_map = v2j_tuple, jumptovars_map = j2v_ranges,
            dep_graph = dg_ranges, alias = true)
        jprob1 = JumpProblem(prob, agg, maj, crj; dependencies = deps)
        deps2 = JumpDependencies(; vartojumps_map = vartojumps_map(),
            jumptovars_map = jumptovars_map(), dep_graph = [[1, 2, 3, 4], [2, 3, 4], [3],
                [2, 3, 4]])
        jprob2 = JumpProblem(prob, agg, maj, crj; dependencies = deps2)
        @test same_solution(jprob1, jprob2)
    end
    # ranges out of normal form can't be changed in place, so the problem builds new maps
    deps = JumpDependencies(; dep_graph = [2:4, 2:4, 3:3, 2:4], alias = true)
    jprob1 = JumpProblem(prob, NRM(), maj, crj; dependencies = deps)
    @test deps.dep_graph == [2:4, 2:4, 3:3, 2:4]
    jprob2 = JumpProblem(prob, NRM(), maj, crj;
        dependencies = JumpDependencies(;
            dep_graph = [[1, 2, 3, 4], [2, 3, 4], [3],
            [2, 3, 4]]))
    @test same_solution(jprob1, jprob2)
end

@testset "JumpDependencies with Coevolve and bounded VariableRateJumps" begin
    # a fifth jump, A → ∅ at a rate rising with t, bounded on [t, t + 1]
    vrj = VariableRateJump((u, p, t) -> u[1] * t / 20, integ -> (integ.u[1] -= 1; nothing);
        urate = (u, p, t) -> u[1] * (t + 1) / 20, rateinterval = (u, p, t) -> 1.0)
    dg5 = [[1, 2, 4, 5], [2, 4, 5], [3], [2, 3, 4, 5], [2, 4, 5]]
    raw_dg5 = [[5, 4, 2], [2, 4, 5, 5], [3], [2, 3, 4, 5], [4, 2]]
    jprob1 = JumpProblem(prob, Coevolve(), maj, crj, vrj;
        dependencies = JumpDependencies(; dep_graph = raw_dg5))
    jprob2 = JumpProblem(prob, Coevolve(), maj, crj, vrj;
        dependencies = JumpDependencies(; dep_graph = dg5))
    @test same_solution(jprob1, jprob2)
    jprob3 = @test_deprecated JumpProblem(prob, Coevolve(), maj, crj, vrj;
        dep_graph = raw_dg5)
    @test same_solution(jprob2, jprob3)
    # the bounded VariableRateJump has an entry, so four entries are too few
    @test_throws ArgumentError JumpProblem(prob, Coevolve(), maj, crj, vrj;
        dependencies = JumpDependencies(; dep_graph = dep_graph()))
end

@testset "Spatial aggregators" begin
    # A + B → C and C → A + B on a line of three sites
    rx_maj = MassActionJump([0.1, 1.0], [[1 => 1, 2 => 1], [3 => 1]],
        [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]])
    u0 = zeros(Int, 3, 3)
    u0[:, 2] .= [50, 50, 0]
    sprob = DiscreteProblem(u0, (0.0, 2.0))
    hopping_constants = fill(1.0, size(u0))
    grid = CartesianGrid((3,))
    raw = JumpDependencies(; vartojumps_map = [[1, 1], [1], [2]],
        jumptovars_map = [[3, 1, 2], [2, 1, 3]], dep_graph = [[2], [2, 1]])
    for agg in (NSM(), DirectCRDirect())
        sjprob(; kwargs...) = JumpProblem(sprob, agg, rx_maj; hopping_constants,
            spatial_system = grid, save_positions = (false, false), kwargs...)
        @test same_solution(sjprob(), sjprob(; dependencies = raw, alias_dependencies = false))
        jprob = @test_deprecated sjprob(; vartojumps_map = [[1], [1], [2]],
            jumptovars_map = [[1, 2, 3], [1, 2, 3]], dep_graph = [[1, 2], [1, 2]])
        @test same_solution(sjprob(), jprob)
    end
end

# with 30 constant rate jumps, the maps decide which aggregator JumpProblem picks
@testset "JumpProblem picks the aggregator from JumpDependencies" begin
    n = 30
    crjs = [ConstantRateJump((u, p, t) -> 1.0, integ -> (integ.u[1] += 1; nothing))
            for _ in 1:n]
    p1 = DiscreteProblem([0], (0.0, 1.0))
    g = [Int[] for _ in 1:n]
    v2j, j2v = [Int[]], [[1] for _ in 1:n]
    jprob = JumpProblem(p1, JumpSet(crjs...);
        dependencies = JumpDependencies(; dep_graph = g))
    @test jprob.aggregator isa SortingDirect
    jprob = @test_deprecated JumpProblem(p1, JumpSet(crjs...); dep_graph = g)
    @test jprob.aggregator isa SortingDirect
    jprob = JumpProblem(p1, JumpSet(crjs...);
        dependencies = JumpDependencies(; vartojumps_map = v2j, jumptovars_map = j2v))
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
    jprob = @test_deprecated JumpProblem(prob, RSSA(), maj, crj; dependencies = deps,
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
        @test_throws ArgumentError JumpProblem(prob, Direct(), maj, crj;
            dependencies = deps)
    end
    # the same maps given as keywords are not checked, as before JumpDependencies
    for maps in bad_maps
        jprob = @test_deprecated JumpProblem(prob, Direct(), maj, crj; maps...)
        @test jprob isa JumpProblem
    end
end

@testset "remake keeps the dependencies" begin
    jprob = JumpProblem(prob, NRM(), maj, crj; dependencies = normal_deps())
    jprob2 = JumpProblem(remake(prob; u0 = [20, 3]), NRM(), maj, crj;
        dependencies = normal_deps())
    @test same_solution(remake(jprob; u0 = [20, 3]), jprob2; seed = 2)
end
