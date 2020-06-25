using DiffEqJump, DiffEqBase, Plots

doplot = false

function plot_solution(sol)
    println("Plotting")
    labels = vcat([["A $i", "B $i", "C $i"] for i in 1:num_nodes]...)
    trajectories = [hcat(sol.u...)[i,:] for i in 1:length(spatial_jump_prob.prob.u0)]
    plot1 = plot(sol.t, trajectories[1], label = labels[1])
    for i in 2:3
        plot!(plot1, sol.t, trajectories[i], label = labels[i])
    end
    title!("A + B <--> C RDME")
    xaxis!("time")
    yaxis!("number")
    plot1
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
spec_to_dep_jumps = [[1],[1],[2]]
jump_to_dep_specs = [[1,2,3],[1,2,3]]
rates = [0.1, 1.]
majumps = MassActionJump(rates, reactstoch, netstoch)
prob = DiscreteProblem([100,100,0],(0.0,0.25), rates)

# Graph setup
domain_size = 1.0 #μ-meter
num_sites_per_edge = 20
diffusivity = 0.1
hopping_rate = diffusivity * (num_sites_per_edge/domain_size)^2
dimension = 2
connectivity_list = connectivity_list_from_box(num_sites_per_edge, dimension)
num_nodes = length(connectivity_list)

diff_rates_for_edge = [hopping_rate for species in 1:length(prob.u0)]
diff_rates = [[diff_rates_for_edge for j in 1:length(connectivity_list[i])] for i in 1:num_nodes]

# Solving
alg = RSSACR()
println("Solving with $alg")
jump_prob = JumpProblem(prob, alg, majumps, save_positions=(false,false), vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
starting_state = zeros(Integer, num_nodes*length(prob.u0))
# starting_state[1 : length(prob.u0)] = copy(prob.u0)
center_node = coordinates_to_node(trunc(Integer,num_sites_per_edge/2),trunc(Integer,num_sites_per_edge/2),num_sites_per_edge)
center_node_first_species_index = to_spatial_spec(center_node, 1, length(prob.u0))
starting_state[center_node_first_species_index : center_node_first_species_index + length(prob.u0) - 1] = copy(prob.u0)
spatial_jump_prob = to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob, starting_state = starting_state)
sol = solve(spatial_jump_prob, SSAStepper(), saveat = prob.tspan[2]/50)

if doplot
    # Plotting
    plt = plot_solution(sol)
    display(plt)
end

# Make animation
"get frame k"
function get_frame(k,sol, num_species, num_sites_per_edge)
    times = sol.t
    states = sol.u
    h = 1/num_sites_per_edge
    t = times[k]
    state = states[k]
    plt = plot(xlim=(0,1), ylim=(0,1), title = "A + B <--> C, $(round(t, sigdigits=3)) seconds")
    labels = ["A", "B", "C"]

    species_seriess_x = [[] for i in 1:num_species]
    species_seriess_y = [[] for i in 1:num_species]
    for (species_spatial_index, number_of_molecules) in enumerate(state)
        node, species = from_spatial_spec(species_spatial_index, num_species)
        x,y,_ = node_to_coordinates(node, num_sites_per_edge)
        for k in 1:number_of_molecules
            push!(species_seriess_x[species], x*h - h/2 + h*rand())
            push!(species_seriess_y[species], y*h - h/2 + h*rand())
        end
    end
    for species in 1:num_species
        scatter!(plt, species_seriess_x[species], species_seriess_y[species], label = labels[species], marker = 3)
    end
    xticks!(plt, range(0,1,length = num_sites_per_edge+1))
    yticks!(plt, range(0,1,length = num_sites_per_edge+1))
    xgrid!(plt, 1, 0.7)
    ygrid!(plt, 1, 0.7)
    return plt
end

anim = @animate for k=1:length(sol.t)
    get_frame(k, sol, length(prob.u0), num_sites_per_edge)
    println("Done with frame $k")
end

gif(anim, "test/spatial/anim.gif", fps = 2)
