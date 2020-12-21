using DiffEqBase, DiffEqJump, DataStructures
using Test, Statistics

# using Plots; plotlyjs()
doplot = false

# using BenchmarkTools
# dobenchmark = false

dotestmean   = true
doprintmeans = false

# SSAs to test
SSAalgs = (RDirect(), RSSACR(),Direct(), DirectFW(), FRM(), FRMFW(), SortingDirect(), NRM(), RSSA(), DirectCR())

Nsims        = 32000
tf           = .01
u0           = [200, 100, 150]
expected_avg = 84.876015624999994
reltol       = .01

# MODEL SETUP

# using DiffEqBiological
# rs = @reaction_network dtype begin
#     k1, 2A --> B
#     k2, B --> 2A
#     k3, A + B --> C
#     k4, C --> A + B
#     k5, 3C --> 3A
# end k1 k2 k3 k4

# model using mass action jumps
# ids: A = 1, B = 2, C = 3
reactstoch = [
    [1 => 2],
    [2 => 1],
    [1 => 1, 2 => 1],
    [3 => 1],
    [3 => 3]
]
netstoch = [
    [1 => -2, 2 => 1],
    [1 => 2, 2 => -1],
    [1 => -1, 2 => -1, 3 => 1],
    [1 => 1, 2 => 1, 3 => -1],
    [1 => 3, 3 => -3]
]
rates = [1., 2., .5, .75, .25]
spec_to_dep_jumps = [[1,3],[2,3],[4,5]]
jump_to_dep_specs = [[1,2],[1,2],[1,2,3],[1,2,3],[1,3]]
majumps = MassActionJump(rates, reactstoch, netstoch)

# average number of proteins in a simulation
function runSSAs(jump_prob)
    Psamp = zeros(Int, Nsims)
    for i in 1:Nsims
        sol = solve(jump_prob, SSAStepper())
        Psamp[i] = sol[1,end]
    end
    mean(Psamp)
end

# TESTING:
prob = DiscreteProblem(u0, (0.0, tf), rates)

# plotting one full trajectory
if doplot
    for alg in SSAalgs
        local jump_prob = JumpProblem(prob, alg, majumps, vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
        local sol = solve(jump_prob, SSAStepper())
        local plothand = plot(sol, seriestype=:steppost, reuse=false)
        display(plothand)
    end
end

# test the means
if dotestmean
    means = zeros(Float64,length(SSAalgs))
    for (i,alg) in enumerate(SSAalgs)
        local jump_prob = JumpProblem(prob, alg, majumps, save_positions=(false,false), vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
        means[i]  = runSSAs(jump_prob)
        relerr = abs(means[i] - expected_avg) / expected_avg
        if doprintmeans
            println("Mean from method: ", typeof(alg), " is = ", means[i], ", rel err = ", relerr)
        end

        # if dobenchmark
        #      @btime (runSSAs($jump_prob);)
        # end


        @test abs(means[i] - expected_avg) < reltol*expected_avg
    end
end


# benchmark performance
# if dobenchmark
#     # exact methods
#     for alg in SSAalgs
#         println("Solving with method: ", typeof(alg), ", using SSAStepper")
#         jump_prob = JumpProblem(prob, alg, majumps, vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
#         @btime solve($jump_prob, SSAStepper())
#     end
#     println()
# end



# add a test for passing MassActionJumps individually (tests combining)
if dotestmean
    majump_vec = Vector{MassActionJump{Float64,Vector{Pair{Int,Int}}}}()
    for i = 1:length(rates)
        push!(majump_vec, MassActionJump(rates[i], reactstoch[i], netstoch[i]))
    end
    jset = JumpSet((),(),nothing,majump_vec)
    jump_prob = JumpProblem(prob, Direct(), jset, save_positions=(false,false), vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
    meanval = runSSAs(jump_prob)
    relerr = abs(meanval - expected_avg) / expected_avg
    if doprintmeans
        println("Using individual MassActionJumps; Mean from method: ", typeof(Direct()), " is = ", meanval, ", rel err = ", relerr)
    end
    @test abs(meanval - expected_avg) < reltol*expected_avg
end
