using DiffEqJump, DiffEqBase, Parameters, Plots
using LightGraphs, BenchmarkTools

doplot = true
dobenchmark = false

# functions to specify reactions between neighboring nodes
"given a multimolecular reaction, assign its products to the source and the target"
function assign_products(rx, massaction_jump, (node1, species1), (node2, species2))
    if species1 == 2
        return [2 => 2], convert(typeof([2 => 2]), [])
    else
        return convert(typeof([2 => 2]), []), [2 => 2]
    end
end

"given a multimolecular reaction, get the rate"
function get_rate(rx, (source, source_species), (target, target_species), rates)
    rates[rx]
end

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
prob = DiscreteProblem([999,1,0],(0.0,250.0), rates)

# Graph setup for SIR model
num_nodes = 3
# NOTE: to change the graph, change connectivity_list
connectivity_list = [[mod1(i-1,num_nodes),mod1(i+1,num_nodes)] for i in 1:num_nodes] # this is a cycle graph

# diff_rate = 0.1

if doplot
    # Solve and plot: neighbors not reacting
    alg = WellMixedSpatial(RDirect())
    spatial_SIR = JumpProblem(prob, alg, majumps, connectivity_list)
    println("Solving...")
    sol = solve(spatial_SIR, SSAStepper(), saveat = 1.)
    println("Plotting...")
    labels = vcat([["S $i", "I $i", "R $i"] for i in 1:num_nodes]...)
    trajectories = [hcat(sol.u...)[i,:] for i in 1:3*num_nodes]
    plot1 = plot(sol.t, trajectories[1], label = labels[1])
    for i in 2:length(trajectories)
        plot!(plot1, sol.t, trajectories[i], label = labels[i])
    end
    title!("SIR with neighbors not allowed to react")
    xaxis!("time")
    yaxis!("number")

    # Solve and plot: neighbors reacting
    spatial_SIR = JumpProblem(prob, alg, majumps, connectivity_list; assign_products = assign_products)
    sol = solve(spatial_SIR, SSAStepper(), saveat = 1.)
    labels = vcat([["S $i", "I $i", "R $i"] for i in 1:num_nodes]...)
    trajectories = [hcat(sol.u...)[i,:] for i in 1:3*num_nodes]
    plot2 = plot(sol.t, trajectories[1], label = labels[1])
    for i in 2:length(trajectories)
        plot!(plot2, sol.t, trajectories[i], label = labels[i])
    end
    title!("SIR with neighbors allowed to react")
    xaxis!("time")
    yaxis!("number")

    layout = @layout [a ; b]
    final_plot = plot(plot1, plot2, layout = layout)
    println("Displaying the plot:")
    display(final_plot)
end

################# Benchmarking ################
function benchmark_n_times(jump_prob, n)
    @elapsed solve(jump_prob, SSAStepper(), saveat = 10.)
    times = []
    for i in 1:n
        push!(times, @elapsed solve(jump_prob, SSAStepper(), saveat = 10.))
    end
    times
end

function run_n_steps(n, p, integrator)
    for i in 1:n
        p(integrator)
    end
    nothing
end

"construct the connectivity list of light graph g"
function get_connectivity_list(g)
    a = adjacency_matrix(g)
    [[c for c in 1:size(a)[2] if a[r,c] != zero(a[1])] for r in 1:size(a)[1]]
end
if dobenchmark

    num_nodes = 500
    println("Have $num_nodes nodes.")
    g = random_regular_digraph(num_nodes, 5)
    connectivity_list = get_connectivity_list(g)

    diff_rates_for_edge = Array{Float64,1}(undef,length(jump_prob_SIR.prob.u0))
    diff_rates_for_edge[1] = 0.01
    diff_rates_for_edge[2] = 0.01
    diff_rates_for_edge[3] = 0.01
    diff_rates = [[diff_rates_for_edge for j in 1:length(connectivity_list[i])] for i in 1:num_nodes]
    println("Starting benchmark")
    for alg in [RSSACR(), DirectCR()]
        spatial_SIR = JumpProblem(prob, WellMixedSpatial(alg), majumps, connectivity_list; assign_products = assign_products)
        println("Using $(spatial_SIR.aggregator)")
        median_time = median(benchmark_n_times(spatial_SIR, 5))
        println("Solving the problem took $median_time seconds.")
        @btime solve($spatial_SIR, $(SSAStepper()))

        integrator = init(spatial_SIR, SSAStepper())
        p = spatial_SIR.discrete_jump_aggregation;

        init_allocs = @allocated p(0, integrator.u, integrator.t, integrator)
        run_allocs = @allocated run_n_steps(10^6, p, integrator)
        println("$(spatial_SIR.aggregator) allocated $init_allocs bytes to initialize and $run_allocs bytes to run 10^6 steps")

    end
end

################### Allocation tests #######################


# num_nodes = 500
# g = random_regular_digraph(num_nodes, 5)
# connectivity_list = get_connectivity_list(g)
# diff_rates_for_edge = Array{Float64,1}(undef,length(jump_prob_SIR.prob.u0))
# diff_rates_for_edge[1] = 0.01
# diff_rates_for_edge[2] = 0.01
# diff_rates_for_edge[3] = 0.01
# diff_rates = [[diff_rates_for_edge for j in 1:length(connectivity_list[i])] for i in 1:num_nodes]
#
# function test_allocs(alg, spatial_SIR, integrator, p, n)
#     solve(spatial_SIR, SSAStepper())
#     p(0, integrator.u, integrator.t, integrator)
#     init_allocs = @allocated p(0, integrator.u, integrator.t, integrator)
#     run_n_steps(n, p, integrator)
#     run_allocs = @allocated run_n_steps(n, p, integrator)
#     println("$(spatial_SIR.aggregator) allocated $init_allocs bytes to initialize and $run_allocs bytes to run $n steps. This is $(run_allocs/n) allocations per step")
#     println("Running $n steps:")
#     @btime run_n_steps(n, p, integrator)
#     # println("Solving:")
#     # @btime solve($spatial_SIR, $(SSAStepper()))
# end
#
# n = 10^5
# for alg in [DirectCR(), RSSACR()]
#     spatial_SIR = to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob_SIR, alg, assign_products, get_rate);
#     integrator = init(spatial_SIR, SSAStepper());
#     p = spatial_SIR.discrete_jump_aggregation;
#     test_allocs(alg, spatial_SIR, integrator, p, n)
# end

# println(@allocated DiffEqJump.execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t))
# println(@allocated DiffEqJump.generate_jumps!(p, integrator, integrator.u, integrator.p, integrator.t))
# println(@allocated DiffEqJump.register_next_jump_time!(integrator, p, integrator.t))


# println(@allocated DiffEqJump.execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t))
# println(@allocated DiffEqJump.generate_jumps!(p, integrator.u, integrator.p, integrator.t))
# println(@allocated DiffEqJump.register_next_jump_time!(integrator, p, integrator.t))

# @btime DiffEqJump.execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
# @btime DiffEqJump.generate_jumps!(p, integrator.u, integrator.p, integrator.t)
# @btime DiffEqJump.register_next_jump_time!(integrator, p, integrator.t)
#
# @code_warntype DiffEqJump.execute_jumps!(p, integrator, integrator.u, integrator.p, integrator.t)
# @code_warntype DiffEqJump.generate_jumps!(p, integrator.u, integrator.p, integrator.t)
# @code_warntype DiffEqJump.register_next_jump_time!(integrator, p, integrator.t)
#
