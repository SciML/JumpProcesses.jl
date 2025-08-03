using Test, JumpProcesses, StableRNGs

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
        jprob = JumpProblem(dprob, agg, maj; save_positions = (false, false), rng,
            vartojumps_map = vtoj, jumptovars_map = jtov, dep_graph = dg,
            scale_rates = false)
        for i in 1:Nsims
            sol = solve(jprob, SSAStepper())
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
