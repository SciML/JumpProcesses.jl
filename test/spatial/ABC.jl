using DiffEqJump, DiffEqBase
using Plots, BenchmarkTools
using Test

dospatialtest = true
domixedtest = false
# doplot = false
dobenchmark = false
doanimation = false

function get_mean_end_state(jump_prob, Nsims)
    end_state = zeros(1:length(jump_prob.prob.u0))
    for i in 1:Nsims
        sol = solve(jump_prob, SSAStepper())
        end_state += sol.u[end]
    end
    end_state/Nsims
end

function plot_solution(sol, nodes)
    println("Plotting")
    labels = vcat([["A $i", "B $i", "C $i"] for i in 1:num_nodes]...)
    trajectories = [hcat(sol.u...)[i,:] for i in 1:length(spatial_jump_prob.prob.u0)]
    p = plot()
    for node in nodes
        for species in 1:3
            plot!(p, sol.t, trajectories[3*(node-1)+species], label = labels[3*(node-1)+species])
        end
    end
    title!("A + B <--> C RDME")
    xaxis!("time")
    yaxis!("number")
    p
end

function benchmark_n_times(jump_prob, n)
    solve(jump_prob, SSAStepper())
    times = zeros(n)
    for i in 1:n
        times[i] = @elapsed solve(jump_prob, SSAStepper())
    end
    times
end

# ABC model A + B <--> C
reactstoch = [
    [1 => 1, 2 => 1],
    [3 => 1],
]
netstoch = [
    [1 => -1, 2 => -1, 3 => 1],
    [1 => 1, 2 => 1, 3 => -1]
]
rates = [0.1, 1.]
majumps = MassActionJump(rates, reactstoch, netstoch)
prob = DiscreteProblem([500,500,0],(0.0,10.0), rates)

function test_alg(jump_prob, Nsims, equilibrium_state)
    diff = get_mean_end_state(jump_prob, Nsims) - equilibrium_state
    println("max relative error: $(maximum(abs.(diff./equilibrium_state)))")
    return diff
end
if domixedtest
    Nsims = 10^5

    alg = RSSACR()
    jump_prob = JumpProblem(prob, alg, majumps, vartojumps_map = DiffEqJump.spec_to_dep_rxs_map(length(prob.u0), majumps), jumptovars_map = DiffEqJump.rxs_to_dep_spec_map(majumps))
    N = sum(starting_state)
    k = rates[2]/rates[1] * 1/2
    A = B = -k + √(k^2 + N*k)
    C = N/2 - A
    equilibrium_state = [A, B, C]
    diff = test_alg(jump_prob, Nsims, equilibrium_state)
end
# Graph setup
domain_size = 1.0 #μ-meter
num_sites_per_edge = 10
diffusivity = 0.1
hopping_rate = diffusivity * (num_sites_per_edge/domain_size)^2
dimension = 1
connectivity_list = connectivity_list_from_box(num_sites_per_edge, dimension) # this is a grid graph
num_nodes = length(connectivity_list)

# Starting state setup
starting_state = zeros(Int, num_nodes*length(prob.u0))
# starting_state[1 : length(prob.u0)] = copy(prob.u0)
center_node = coordinates_to_node(trunc(Int,num_sites_per_edge/2),num_sites_per_edge)
center_node_first_species_index = to_spatial_spec(center_node, 1, length(prob.u0))
starting_state[center_node_first_species_index : center_node_first_species_index + length(prob.u0) - 1] = copy(prob.u0)


if dospatialtest
    Nsims        = 10000
    reltol       = 0.05

    N = sum(starting_state)
    k = num_nodes/2 * rates[2]/rates[1]
    A = B = (-k + √(k^2 + N*k))/num_nodes
    C = N/(2*num_nodes) - A
    equilibrium_state = vcat([[A, B, C] for node in 1:num_nodes]...)

    alg = WellMixedSpatial(RSSACR())
    spatial_jump_prob = JumpProblem(prob, alg, majumps; connectivity_list = connectivity_list, diff_rates = hopping_rate, starting_state = starting_state)
    diff = test_alg(spatial_jump_prob, Nsims, equilibrium_state)
    @test [abs(d) < reltol*equilibrium_state[i] for (i,d) in enumerate(diff)] == [true for d in diff]
end

# if doplot
#     # Solving
#     alg = WellMixedSpatial(RSSACR())
#     println("Solving with $alg")
#     spatial_jump_prob = JumpProblem(prob, alg, majumps; connectivity_list = connectivity_list, diff_rates = hopping_rate, starting_state = starting_state)
#     sol = solve(spatial_jump_prob, SSAStepper(), saveat = prob.tspan[2]/50)
#     # Plotting
#     nodes = [1,8]
#     plt = plot_solution(sol, nodes)
#     display(plt)
# end


if dobenchmark
    rates = [0.1, 1.]
    majumps = MassActionJump(rates, reactstoch, netstoch)
    u0 = [500,500,0]
    tf = 0.1
    prob = DiscreteProblem(u0,(0.0,tf), rates)
    dimension = 3
    hopping_rate = 1. # NOTE: using constant hopping rate not prevent diffusions from skyrocketing as number of sites per edge increases
    println("Using constant hopping rate not prevent diffusions from skyrocketing as number of sites per edge increases")
    println("rates: $rates, hopping rate: $hopping_rate, dimension: $dimension, starting position: $u0, final time: $tf")
    nums_sites_per_edge = [8, 16, 32]
    for (i, num_sites_per_edge) in enumerate(nums_sites_per_edge)
        num_nodes = num_sites_per_edge^3
        num_species = length(prob.u0)
        num_rxs = num_species*6*num_nodes + num_nodes * get_num_majumps(majumps)
        println("Have $num_sites_per_edge sites per edge, $num_nodes nodes and $num_rxs reactions")

        connectivity_list = connectivity_list_from_box(num_sites_per_edge, dimension)

        # Starting state setup (place all in the center)
        starting_state = zeros(Int, num_nodes*length(prob.u0))
        center_node = coordinates_to_node(trunc(Int,num_sites_per_edge/2),trunc(Int,num_sites_per_edge/2),num_sites_per_edge)
        center_node_first_species_index = to_spatial_spec(center_node, 1, length(prob.u0))
        starting_state[center_node_first_species_index : center_node_first_species_index + length(prob.u0) - 1] = copy(prob.u0)

        for alg in [RSSACR(), DirectCR(), NRM()]
            short_label = "$alg"[1:end-2]
            spatial_jump_prob = JumpProblem(prob, alg, majumps; connectivity_list = connectivity_list, diff_rates = hopping_rate, starting_state = starting_state, save_positions=(false,false))
            println("Solving with $(spatial_jump_prob.aggregator)")
            solve(spatial_jump_prob, SSAStepper())
            # times = benchmark_n_times(spatial_jump_prob, 5)
            # median_time = median(times)
            # println("Solving the problem took $median_time seconds.")
            @btime solve($spatial_jump_prob, $(SSAStepper()))
        end
    end
end

# quick benchmark:
# algs = [RSSA(), RSSACR(), NRM(), SortingDirect(), RDirect(), Direct()]
# using BenchmarkTools
# for alg in algs
#     spatial_jump_prob = JumpProblem(prob, WellMixedSpatial(alg), majumps; connectivity_list = connectivity_list, diff_rates = hopping_rate, starting_state = starting_state)
#     println("Solving with $(spatial_jump_prob.aggregator)")
#     solve(spatial_jump_prob, SSAStepper())
#     @time solve(spatial_jump_prob, SSAStepper())
# end

# Make animation
if doanimation
    alg = RSSACR()
    println("Setting up...")
    spatial_jump_prob = JumpProblem(prob, alg, majumps; connectivity_list = connectivity_list, diff_rates = hopping_rate, starting_state = starting_state)
    println("Solving...")
    sol = solve(spatial_jump_prob, SSAStepper(), saveat = prob.tspan[2]/200)
    println("Animating...")
    anim=animate_2d(sol, num_sites_per_edge, species_labels = ["A", "B", "C"], title = "A + B <--> C", verbose = true)
    fps = 15
    path = joinpath(@__DIR__, "test", "spatial", "ABC_anim_$(length(sol.u))frames_$(fps)fps.gif")
    gif(anim, path, fps = fps)
end
