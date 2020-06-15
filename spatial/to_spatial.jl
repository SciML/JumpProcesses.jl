"Given a graph and a jump problem, turn all reactions into diffusions, and output the spatial jump problem"
# Suppose the graph is given as an adjacency list
# Further suppose that for each edge and for each species there is a diffusion constant/rate

using DiffEqJump, DiffEqBase
# Jump problem setup
tf           = 1000.0
u0           = [1,0,0,0]

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
prob = DiscreteProblem(u0, (0.0, tf), rates)
jump_prob = JumpProblem(prob, NRM(), majumps, vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)

# Graph setup
num_nodes = 3
connectivity_list = [[mod1(i-1,num_nodes),mod1(i+1,num_nodes)] for i in 1:num_nodes] # this is a cycle graph

diff_rates_for_edge = Array{Float64,1}(undef,length(jump_prob.prob.u0))
diff_rates_for_edge[1] = 0.01
diff_rates_for_edge[2] = 0.01
diff_rates_for_edge[3] = 1.0
diff_rates_for_edge[4] = 1.0
diff_rates = [[diff_rates_for_edge for j in 1:length(connectivity_list[i])] for i in 1:num_nodes]

################## SET UP SPATIAL PROBLEM ################

"""
Construct a spatial jump problem, where diffusions are represented as reactions given an adjacency/connectivity list, the diffusion rates (analagous to reaction rates) and a non-spatial jump problem

The ordering of species is:
[species in node 1, species in node 2, ... ] <-- lengths = num_species, num_species, ...

The ordering of reactions is:
[instances of reaction 1, instances of reaction 2, ..., <-- lengths = num_nodes, num_nodes, ...
diffusions of species 1, diffusions of species 2, ...   <-- lengths = sum_degrees, sum_degrees, ...]

diffusions of species i = diffusions from node 1, diffusions from node 2, ... <-- lengths = degree of node 1, degree of node 2, ...
"""
function to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob; assign_products = nothing)
    num_nodes = length(connectivity_list)
    sum_degrees = (length ∘ vec ∘ hcat)(connectivity_list...)
    massaction_jump = jump_prob.massaction_jump
    num_majumps = get_num_majumps(massaction_jump)
    num_species = length(jump_prob.prob.u0)
    prob = jump_prob.prob

    # spatial constants
    num_spacial_majumps = num_nodes * num_majumps
    num_spatial_species = num_species*num_nodes
    source_target_index_pairs = get_source_target_index_pairs(connectivity_list, sum_degrees)
    num_spatial_rxs = num_spacial_majumps + num_species*sum_degrees

    # NOTE: only mass action jumps
    rx_rates = Array{Float64,1}(undef, num_spatial_rxs)
    reaction_stoichiometries = Array{Array{Pair{Int64,Int64},1},1}(undef, num_spatial_rxs)
    net_stoichiometries = Array{Array{Pair{Int64,Int64},1},1}(undef, num_spatial_rxs)
    spatial_spec_to_dep_jumps = [Array{Int64,1}(undef,0) for spec in 1:num_spatial_species]
    spatial_jump_to_dep_specs = Array{Array{Int64,1},1}(undef, num_spatial_rxs)

    # make stoichiometries for reactions
    for (i, (node, rx)) in enumerate(Iterators.product(1:num_nodes, 1:num_majumps))
        rx_rates[i] = prob.p[rx]
        reaction_stoichiometries[i], net_stoichiometries[i] = get_rx_stoichiometries(node, rx, massaction_jump, num_species)
        spatial_jump_to_dep_specs[i] = [s for (s, c) in net_stoichiometries[i]]
        for (s,c) in reaction_stoichiometries[i]
            push!(spatial_spec_to_dep_jumps[s], i)
        end
    end
    # make stoichiometries for diffusions
    for (i, ((source, target_index), species)) in enumerate(Iterators.product(source_target_index_pairs, 1:num_species))
        rx_rates[i+num_spacial_majumps] = diff_rates[source][target_index][species]
        reaction_stoichiometries[i+num_spacial_majumps], net_stoichiometries[i+num_spacial_majumps] = get_diff_stoichiometries(source, connectivity_list[source][target_index], species, connectivity_list, num_species)
        spatial_jump_to_dep_specs[i+num_spacial_majumps] = [s for (s, c) in net_stoichiometries[i+num_spacial_majumps]]
        for (s,c) in reaction_stoichiometries[i+num_spacial_majumps]
            push!(spatial_spec_to_dep_jumps[s], i+num_spacial_majumps)
        end
    end

    spatial_majumps = MassActionJump(rx_rates, reaction_stoichiometries, net_stoichiometries)

    # NOTE: arbitrary decision to copy the original u0
    starting_state = vec(hcat([prob.u0 for i in 1:num_nodes]...))
    spatial_prob = DiscreteProblem(starting_state, prob.tspan, rx_rates)
    JumpProblem(spatial_prob, jump_prob.aggregator, spatial_majumps, save_positions=(false,false), vartojumps_map=spatial_spec_to_dep_jumps, jumptovars_map=spatial_jump_to_dep_specs)
end

############## Allow reactions between neighboring nodes ##############
"given a spatial problem that is an output of to_spatial_jump_prob(), add reactions between neighboring nodes"
function add_neighbor_reactions(spatial_jump_prob, assign_products)
    for (i, ((source, target_index), rx)) in enumerate(Iterators.product(source_target_index_pairs, 1:num_majumps))
        rx_rates[i+num_spacial_majumps] = diff_rates[source][target_index][species]
        reaction_stoichiometries[i+num_spacial_majumps], net_stoichiometries[i+num_spacial_majumps] = get_diff_stoichiometries(source, connectivity_list[source][target_index], species, connectivity_list, num_species)
        spatial_jump_to_dep_specs[i+num_spacial_majumps] = [s for (s, c) in net_stoichiometries[i+num_spacial_majumps]]
        for (s,c) in reaction_stoichiometries[i+num_spacial_majumps]
            push!(spatial_spec_to_dep_jumps[s], i+num_spacial_majumps)
        end
    end
end
############ Helper functions ###############
"given a bimolecular reaction, assign its products to the source and the target"
function assign_products(rx, full_net_stoichiometry, (source, source_species), (target, target_species))
    products = [s => c for (s, c) in full_net_stoichiometry if c > 0]
    return source => convert(typeof(full_net_stoichiometry), products), target => convert(typeof(full_net_stoichiometry), [])
end

"given a spatial index, get (node index, original species index)."
function from_spatial_spec(ind, num_species)
    fldmod1(ind, num_species)
end

"get the sptial index of the species in node"
function to_spatial_spec(node, ind, num_species)
    return (node-1)*num_species + ind
end

"get the stoichiometries for reaction rx (non-spatial index) in node"
function get_rx_stoichiometries(node, rx, massaction_jump, num_species)
    [Pair(to_spatial_spec(node, ind, num_species), coeff) for (ind, coeff) in massaction_jump.reactant_stoch[rx]],
    [Pair(to_spatial_spec(node, ind, num_species), coeff) for (ind, coeff) in massaction_jump.net_stoch[rx]]
end

"get the stoichiometries for diffusion of species (non-spatial index) from source to target"
function get_diff_stoichiometries(source, target, species, connectivity_list, num_species)
    [Pair(to_spatial_spec(source, species, num_species), 1)],
    [Pair(to_spatial_spec(source, species, num_species),-1), Pair(to_spatial_spec(target, species, num_species),1)]
end

"get all source-target_index pairs of nodes"
function get_source_target_index_pairs(connectivity_list, sum_degrees)
    vec(hcat([[Pair(source, target_index) for target_index in 1:length(connectivity_list[source])] for source in 1:length(connectivity_list)]...))
end
