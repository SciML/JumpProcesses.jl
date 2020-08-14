using DiffEqJump, DiffEqBase, Parameters
# using Plots
# using LightGraphs, BenchmarkTools
using Test

doplot = false
dobenchmark = false

function get_mean_sol(jump_prob, Nsims, saveat)
    sol = solve(jump_prob, SSAStepper(), saveat = saveat).u
    for i in 1:Nsims-1
        sol += solve(jump_prob, SSAStepper(), saveat = saveat).u
    end
    sol/Nsims
end

# functions to specify reactions between neighboring nodes
"given a multimolecular reaction, assign its products to the source and the target"
function assign_products(rx, massaction_jump, (node1, species1), (node2, species2))
    if species1 == 2
        return [2 => 2], convert(typeof([2 => 2]), [])
    else
        return convert(typeof([2 => 2]), []), [2 => 2]
    end
end

# NOTE: this is the default get_rate:
# "given a multimolecular reaction, get the rate"
# function get_rate(rx, (node1, species1), (node2, species2), rates)
#     rates[rx]
# end

# SIR model
reactstoch = [
    [1 => 1, 2 => 1],
    [2 => 1],
]
netstoch = [
    [1 => -1, 2 => 1],
    [2 => -1, 3 => 1],
]
spec_to_dep_jumps = [[1],[1,2],convert(Array{Int64,1}, [])]
jump_to_dep_specs = [[1,2],[2,3]]
rates = [1e-4, 0.01]
majumps = MassActionJump(rates, reactstoch, netstoch)
u0 = [998,2,0]

# Graph setup for SIR model
num_nodes = 3
connectivity_list = [[mod1(i-1,num_nodes),mod1(i+1,num_nodes)] for i in 1:num_nodes] # this is a cycle graph
starting_state = vcat([u0 for i in 1:num_nodes]...)
tf = 250.
prob = DiscreteProblem(starting_state,(0.0,tf), rates)

alg = WellMixedSpatial(RDirect())
jprob = JumpProblem(prob, alg, majumps, connectivity_list = connectivity_list)
jprob_neighbors_react = JumpProblem(prob, alg, majumps, connectivity_list = connectivity_list, assign_products = assign_products)
sol1 = get_mean_sol(jprob, 100, 1.)
sol2 = get_mean_sol(jprob_neighbors_react, 100, 1.)
trajectories1 = [hcat(sol1...)[i,:] for i in 1:3*num_nodes]
trajectories2 = [hcat(sol2...)[i,:] for i in 1:3*num_nodes]

# testing that if neighbors react people get infected faster.
@test sum(sol1[45][2:3:end]) < 0.5 * sum(sol2[45][2:3:end])


############### PLOTTING ###############
# if doplot
#     Nsims = 100
#     alg = WellMixedSpatial(RDirect())
#     jprob = JumpProblem(prob, alg, majumps, connectivity_list = connectivity_list)
#     jprob_neighbors_react = JumpProblem(prob, alg, majumps, connectivity_list = connectivity_list, assign_products = assign_products)
#     sol1 = get_mean_sol(jprob, Nsims, 1.)
#     sol2 = get_mean_sol(jprob_neighbors_react, Nsims, 1.)
#
#     labels = vcat([["S $i", "I $i", "R $i"] for i in 1:num_nodes]...)
#     trajectories1 = [hcat(sol1...)[i,:] for i in 1:3*num_nodes]
#     plot1 = plot()
#     for i in 1:length(trajectories)
#         plot!(plot1, 0:1:tf, trajectories1[i], label = labels[i])
#     end
#     title!("SIR with neighbors not allowed to react")
#     xaxis!("time")
#     yaxis!("number")
#
#     trajectories2 = [hcat(sol2...)[i,:] for i in 1:3*num_nodes]
#     plot2 = plot()
#     for i in 1:length(trajectories)
#         plot!(plot2, 0:1:tf, trajectories2[i], label = labels[i])
#     end
#     title!("SIR with neighbors not allowed to react")
#     xaxis!("time")
#     yaxis!("number")
#
#     layout = @layout [a ; b]
#     final_plot = plot(plot1, plot2, layout = layout)
#     display(final_plot)
#
# end


################# Benchmarking ################
# function benchmark_n_times(jump_prob, n)
#     @elapsed solve(jump_prob, SSAStepper(), saveat = 10.)
#     times = []
#     for i in 1:n
#         push!(times, @elapsed solve(jump_prob, SSAStepper(), saveat = 10.))
#     end
#     times
# end
#
# function run_n_steps(n, p, integrator)
#     for i in 1:n
#         p(integrator)
#     end
#     nothing
# end
#
# "construct the connectivity list of light graph g"
# function get_connectivity_list(g)
#     a = adjacency_matrix(g)
#     [[c for c in 1:size(a)[2] if a[r,c] != zero(a[1])] for r in 1:size(a)[1]]
# end
# if dobenchmark
#
#     num_nodes = 500
#     println("Have $num_nodes nodes.")
#     g = random_regular_digraph(num_nodes, 5)
#     connectivity_list = get_connectivity_list(g)
#
#     diff_rates_for_edge = Array{Float64,1}(undef,length(jump_prob_SIR.prob.u0))
#     diff_rates_for_edge[1] = 0.01
#     diff_rates_for_edge[2] = 0.01
#     diff_rates_for_edge[3] = 0.01
#     diff_rates = [[diff_rates_for_edge for j in 1:length(connectivity_list[i])] for i in 1:num_nodes]
#     println("Starting benchmark")
#     for alg in [RSSACR(), DirectCR()]
#         spatial_SIR = JumpProblem(prob, WellMixedSpatial(alg), majumps; connectivity_list = connectivity_list, assign_products = assign_products)
#         println("Using $(spatial_SIR.aggregator)")
#         median_time = median(benchmark_n_times(spatial_SIR, 5))
#         println("Solving the problem took $median_time seconds.")
#         @btime solve($spatial_SIR, $(SSAStepper()))
#
#         integrator = init(spatial_SIR, SSAStepper())
#         p = spatial_SIR.discrete_jump_aggregation;
#
#         init_allocs = @allocated p(0, integrator.u, integrator.t, integrator)
#         run_allocs = @allocated run_n_steps(10^6, p, integrator)
#         println("$(spatial_SIR.aggregator) allocated $init_allocs bytes to initialize and $run_allocs bytes to run 10^6 steps")
#
#     end
# end
