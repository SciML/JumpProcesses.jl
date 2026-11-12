using Test, JumpProcesses, SciMLBase, StableRNGs, ForwardDiff

rng = StableRNG(12345)

# test for https://github.com/SciML/JumpProcesses.jl/issues/479
#    A, ∅ --> X
#    1, 2X + Y --> 3X
#    B, X --> Y
#    1, X --> ∅
function test(rng)
    # dep graphs
    dg = [[1, 2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4]]
    vtoj = [[2, 3, 4],
        [2]]
    jtov = [[1],
        [1, 2],
        [1, 2],
        [1]]

    # reaction as MassActionJump
    sr = [1.0, 0.5, 4.0, 1.0]
    rs = [Pair{Int, Int}[], [1 => 2, 2 => 1], [1 => 1], [1 => 1]]
    ns = [[1 => 1], [1 => 1, 2 => -1], [1 => -1, 2 => 1], [1 => -1]]
    maj = MassActionJump(sr, rs, ns; scale_rates = false)

    Nsims = 10000
    u0 = [1.0, 10.0]
    tspan = (0.0, 50.0)
    dprob = DiscreteProblem(u0, tspan)
    SSAalgs = JumpProcesses.JUMP_AGGREGATORS
    Xmeans = zeros(length(SSAalgs))
    Ymeans = zeros(length(SSAalgs))
    for (j, agg) in enumerate(SSAalgs)
        jprob = JumpProblem(dprob, agg, maj; save_positions = (false, false),
            dependencies = JumpDependencies(; vartojumps_map = vtoj,
                jumptovars_map = jtov, dep_graph = dg))
        for i in 1:Nsims
            sol = solve(jprob, SSAStepper(); rng)
            Xmeans[j] += sol[1, end]
            Ymeans[j] += sol[2, end]
        end
    end
    Xmeans ./= Nsims
    Ymeans ./= Nsims
    # for i in 2:length(SSAalgs)
    #     @test abs(Xmeans[i] - Xmeans[1]) < (.1 * Xmeans[1])
    #     @test abs(Ymeans[i] - Ymeans[1]) < (.1 * Ymeans[1])
    # end
end

# Jump times are drawn in the numeric type of the time (issue #674)
@testset "Exponential draws in the time's type" begin
    typed_randexp = JumpProcesses.typed_randexp
    @test typed_randexp(StableRNG(1), 1.0) == JumpProcesses.randexp(StableRNG(1))
    @test typed_randexp(StableRNG(1), 1.0f0) ==
          Float32(JumpProcesses.randexp(StableRNG(1)))
    for t in (1.0f0, 1.0, big(1.0))
        @test typed_randexp(rng, t) isa typeof(t)
    end
    @test typed_randexp(rng, ForwardDiff.Dual(1.0, 1.0)) isa ForwardDiff.Dual
end

# Float32 and BigFloat times and rates (issue #674): DirectCR, RSSACR, CCNRM and
# DirectCRDirect failed to build or run with them before.
#    10, ∅ --> X
#    1, X --> ∅
# X(10) is close to Poisson(10) distributed when X(0) = 0, so its sample mean is tested to
# within 4 standard deviations, sqrt(10 / Nsims).
@testset "Solves with $T times" for T in (Float32, BigFloat)
    Nsims = (T == Float32) ? 4000 : 1000
    maj = MassActionJump(T[10, 1], [Pair{Int, Int}[], [1 => 1]], [[1 => 1], [1 => -1]])
    dprob = DiscreteProblem([0], (zero(T), T(10)))
    # RDirect can't build problems with BigFloat rates yet
    aggs = (T == BigFloat) ? filter(!=(RDirect()), JumpProcesses.JUMP_AGGREGATORS) :
           JumpProcesses.JUMP_AGGREGATORS
    for agg in aggs
        jprob = JumpProblem(dprob, agg, maj; save_positions = (false, false))
        sol = solve(jprob, SSAStepper(); rng)
        @test eltype(sol.t) == T
        Xmean = sum(solve(jprob, SSAStepper(); rng)[1, end] for _ in 1:Nsims) / Nsims
        @test abs(Xmean - 10) < 4 * sqrt(10 / Nsims)
    end

    # the same reactions at each of two sites, with hopping between them: the total is close
    # to Poisson(20) distributed
    u0 = zeros(Int, 1, 2)
    for agg in (NSM(), DirectCRDirect())
        jprob = JumpProblem(DiscreteProblem(u0, (zero(T), T(10))), agg, maj;
            hopping_constants = fill(one(T), size(u0)), spatial_system = CartesianGrid((2,)),
            save_positions = (false, false))
        sol = solve(jprob, SSAStepper(); rng)
        @test eltype(sol.t) == T
        Xmean = sum(sum(solve(jprob, SSAStepper(); rng).u[end]) for _ in 1:Nsims) / Nsims
        @test abs(Xmean - 20) < 4 * sqrt(20 / Nsims)
    end
end

# Spatial reaction rates are in the time's type: a BigFloat rate below floatmin(Float64)
# rounded to zero, so the reaction never fired.
@testset "Spatial BigFloat rates below floatmin(Float64)" begin
    maj = MassActionJump([big"1e-400"], [[1 => 1]], [[1 => -1]])
    tspan = (big(0.0), big"1e401")
    sol = solve(JumpProblem(DiscreteProblem([1], tspan), Direct(), maj), SSAStepper();
        rng = StableRNG(1))
    for agg in (NSM(), DirectCRDirect())
        jprob = JumpProblem(DiscreteProblem(reshape([1], 1, 1), tspan), agg, maj;
            hopping_constants = zeros(BigFloat, 1, 1), spatial_system = CartesianGrid((1,)))
        ssol = solve(jprob, SSAStepper(); rng = StableRNG(1))
        @test ssol.u[end][1] == 0
        # the first draw gives the same jump time as Direct's
        @test ssol.t[2] == sol.t[2]
    end
end

# Rates at the ends of the numeric range for the rejection (priority) tables, which hung
@testset "Rejection tables with extreme $T rates" for T in (Float32, Float64)
    # a minrate below floatmin(T), given as a Float64
    maj = MassActionJump(T[10, 1], [Pair{Int, Int}[], [1 => 1]], [[1 => 1], [1 => -1]])
    for agg in (DirectCR(), RSSACR())
        jprob = JumpProblem(DiscreteProblem([0], (zero(T), T(10))), agg, maj;
            minrate = 1e-50)
        @test solve(jprob, SSAStepper(); rng).u[end][1] > 0
    end
    jprob = JumpProblem(DiscreteProblem(zeros(Int, 1, 2), (zero(T), T(10))),
        DirectCRDirect(), maj; hopping_constants = ones(T, 1, 2),
        spatial_system = CartesianGrid((2,)), minrate = 1e-50)
    @test sum(solve(jprob, SSAStepper(); rng).u[end]) > 0

    # the largest finite power of two as a rate, until 10 jumps are expected
    rate = T(2)^exponent(floatmax(T))
    maj = MassActionJump([rate], [Pair{Int, Int}[]], [[1 => 1]])
    tspan = (zero(T), 10 / rate)
    for agg in (DirectCR(), RSSACR())
        jprob = JumpProblem(DiscreteProblem([0], tspan), agg, maj;
            save_positions = (false, false))
        @test solve(jprob, SSAStepper(); rng).u[end][1] > 0
    end
    jprob = JumpProblem(DiscreteProblem(zeros(Int, 1, 1), tspan), DirectCRDirect(), maj;
        hopping_constants = zeros(T, 1, 1), spatial_system = CartesianGrid((1,)),
        save_positions = (false, false))
    @test solve(jprob, SSAStepper(); rng).u[end][1] > 0
end
