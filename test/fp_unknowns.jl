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

    # CCNRM's time table holds times of the problem's type
    maj = MassActionJump([10.0f0, 1.0f0], [Pair{Int, Int}[], [1 => 1]], [
        [1 => 1], [1 => -1]])
    dprob = DiscreteProblem([0], (0.0f0, 10.0f0))
    sol = solve(JumpProblem(dprob, CCNRM(), maj), SSAStepper(); rng)
    @test eltype(sol.t) == Float32
end
