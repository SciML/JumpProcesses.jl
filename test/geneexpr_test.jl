using DiffEqBase, JumpProcesses
using Test, Statistics
using StableRNGs
rng = StableRNG(12345)

# using Plots; plotlyjs()
doplot = false
#using BenchmarkTools
# dobenchmark = false

dotestmean = true
doprintmeans = false

# SSAs to test
SSAalgs = (JumpProcesses.JUMP_AGGREGATORS..., JumpProcesses.NullAggregator)

# numerical parameters
Nsims = 8000
tf = 1000.0
u0 = [1, 0, 0, 0]
expected_avg = 5.926553750000000e+02
reltol = 0.01

# average number of proteins in a simulation
function runSSAs(jump_prob; use_stepper = true)
    Psamp = zeros(Int, Nsims)
    for i in 1:Nsims
        sol = use_stepper ? solve(jump_prob, SSAStepper()) : solve(jump_prob)
        Psamp[i] = sol[3, end]
    end
    mean(Psamp)
end

# MODEL SETUP

# DNA repression model DiffEqBiological
# using DiffEqBiological
# rs = @reaction_network dtype begin
#     k1, DNA --> mRNA + DNA
#     k2, mRNA --> mRNA + P
#     k3, mRNA --> 0
#     k4, P --> 0
#     k5, DNA + P --> DNAR
#     k6, DNAR --> DNA + P
# end k1 k2 k3 k4 k5 k6

# model using mass action jumps
# ids: DNA=1, mRNA = 2, P = 3, DNAR = 4
reactstoch = [
    [1 => 1],
    [2 => 1],
    [2 => 1],
    [3 => 1],
    [1 => 1, 3 => 1],
    [4 => 1]
]
netstoch = [
    [2 => 1],
    [3 => 1],
    [2 => -1],
    [3 => -1],
    [1 => -1, 3 => -1, 4 => 1],
    [1 => 1, 3 => 1, 4 => -1]
]
spec_to_dep_jumps = [[1, 5], [2, 3], [4, 5], [6]]
jump_to_dep_specs = [[2], [3], [2], [3], [1, 3, 4], [1, 3, 4]]
rates = [0.5, (20 * log(2.0) / 120.0), (log(2.0) / 120.0), (log(2.0) / 600.0), 0.025, 1.0]
majumps = MassActionJump(rates, reactstoch, netstoch)

# TESTING:
prob = DiscreteProblem(u0, (0.0, tf), rates)

# plotting one full trajectory
if doplot
    plothand = plot(reuse = false)
    for alg in SSAalgs
        local jump_prob = JumpProblem(prob, alg, majumps,
            vartojumps_map = spec_to_dep_jumps,
            jumptovars_map = jump_to_dep_specs, rng = rng)
        local sol = solve(jump_prob, SSAStepper())
        plot!(plothand, sol.t, sol[3, :], seriestype = :steppost)
    end
    display(plothand)
end

# test the means
if dotestmean
    for (i, alg) in enumerate(SSAalgs)
        local jump_prob = JumpProblem(prob, alg, majumps, save_positions = (false, false),
            vartojumps_map = spec_to_dep_jumps,
            jumptovars_map = jump_to_dep_specs, rng = rng)
        means = runSSAs(jump_prob)
        relerr = abs(means - expected_avg) / expected_avg
        doprintmeans && println("Mean from method: ", typeof(alg), " is = ", means,
            ", rel err = ", relerr)
        @test abs(means - expected_avg) < reltol * expected_avg

        means = runSSAs(jump_prob; use_stepper = false)
        relerr = abs(means - expected_avg) / expected_avg
        @test abs(means - expected_avg) < reltol * expected_avg
    end
end

# no-aggregator tests
jump_prob = JumpProblem(prob, majumps; save_positions = (false, false),
    vartojumps_map = spec_to_dep_jumps, jumptovars_map = jump_to_dep_specs, rng)
@test abs(runSSAs(jump_prob) - expected_avg) < reltol * expected_avg
@test abs(runSSAs(jump_prob; use_stepper = false) - expected_avg) < reltol * expected_avg

jump_prob = JumpProblem(prob, majumps, save_positions = (false, false), rng = rng)
@test abs(runSSAs(jump_prob) - expected_avg) < reltol * expected_avg
@test abs(runSSAs(jump_prob; use_stepper = false) - expected_avg) < reltol * expected_avg
