using DiffEqBase, DiffEqJump
using Test, Statistics
using Parameters

using Plots;
using BenchmarkTools

doplot = true
dobenchmark = false
dotestmean   = false
doprintmeans = false

# numerical parameters
Nsims        = 8000
tf           = 1000.0
u0           = [1,0,0,0]
expected_avg = 5.926553750000000e+02
reltol       = .01

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
spec_to_dep_jumps = [[1,5],[2,3],[4,5],[6]]
jump_to_dep_specs = [[2],[3],[2],[3],[1,3,4],[1,3,4]]
rates = [.5, (20*log(2.)/120.), (log(2.)/120.), (log(2.)/600.), .025, 1.]
majumps = MassActionJump(rates, reactstoch, netstoch)

# TESTING:
prob = DiscreteProblem(u0, (0.0, tf), rates)

# SSAs to test
SSAalgs = (Direct(), RSSACR())

function avg_trajectory(jump_prob, Nsims, saveat)
    "get average number of proteins in a simulation"
    @unpack t, u = solve(jump_prob, SSAStepper(),saveat = saveat)
    for i in 1:Nsims-1
        u += solve(jump_prob, SSAStepper(),saveat = saveat).u
    end
    return t, u/Nsims
end

function plot_species(SSAalgs, species)
    pl = plot()
    for alg in SSAalgs
        jump_prob = JumpProblem(prob, alg, majumps, vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs, save_positions=(false,false))
        saveat = jump_prob.prob.tspan[2]/1000.0
        t, u = avg_trajectory(jump_prob, Nsims,saveat)
        plot!(pl, t, [state[species] for state in u], label = ["alg $alg, species $species"])
    end
    display(pl)
end

if doplot
    plot_species(SSAalgs, 3)
end


# test the means
if dotestmean
    means = zeros(Float64,length(SSAalgs))
    for (i,alg) in enumerate(SSAalgs)
        jump_prob = JumpProblem(prob, alg, majumps, save_positions=(false,false), vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
        means[i]  = runSSAs(jump_prob)
        relerr = abs(means[i] - expected_avg) / expected_avg
        if doprintmeans
            println("Mean from method: ", typeof(alg), " is = ", means[i], ", rel err = ", relerr)
        end

        # if dobenchmark
        #     @btime (runSSAs($jump_prob);)
        # end

        @test abs(means[i] - expected_avg) < reltol*expected_avg
    end
end


# benchmark performance
if dobenchmark
    # exact methods
    for alg in SSAalgs
        println("Solving with method: ", typeof(alg), ", using SSAStepper")
        jump_prob = JumpProblem(prob, alg, majumps, vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
        @btime solve($jump_prob, SSAStepper())
    end
    println()
end
