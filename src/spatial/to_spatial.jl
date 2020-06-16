using DiffEqJump, DiffEqBase, Parameters

function to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob)
    to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob, jump_prob.aggregator)
end
function to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob, assign_products, get_rate)
    to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob, jump_prob.aggregator, assign_products, get_rate)
end
"""
Construct a spatial jump problem, where diffusions are represented as reactions.
Given:
adjacency/connectivity list -- representing a graph. Can be directed.
diffusion rates (analagous to reaction rates)
non-spatial jump problem -- only massaction jumps are allowed

The ordering of species is:
[species in node 1, species in node 2, ... ] <-- lengths = num_species, num_species, ...

The ordering of reactions is:
[instances of reaction 1, instances of reaction 2, ..., <-- lengths = num_neighboring_pairs, num_neighboring_pairs, ...
diffusions of species 1, diffusions of species 2, ...   <-- lengths = sum_degrees, sum_degrees, ...]

diffusions of species i = diffusions from node 1, diffusions from node 2, ... <-- lengths = degree of node 1, degree of node 2, ...
"""
function to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob, alg)
    num_nodes = length(connectivity_list)
    sum_degrees = (length ∘ vec ∘ hcat)(connectivity_list...)
    massaction_jump = jump_prob.massaction_jump
    num_majumps = get_num_majumps(massaction_jump)
    num_species = length(jump_prob.prob.u0)
    prob = jump_prob.prob

    # spatial constants
    num_spacial_majumps = num_nodes * num_majumps
    num_spatial_species = num_species*num_nodes
    source_target_index_pairs = get_source_target_index_pairs(connectivity_list)
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
    JumpProblem(spatial_prob, alg, spatial_majumps, save_positions=(false,false), vartojumps_map=spatial_spec_to_dep_jumps, jumptovars_map=spatial_jump_to_dep_specs)
end

############## Allow reactions between neighboring nodes ##############
"""
Construct a jump problem, where diffusions are represented as reactions, and species in neighboring nodes can react.
Given:
adjacency/connectivity list -- representing a graph. Can be directed.
diffusion rates (analagous to reaction rates)
non-spatial jump problem -- only unimolecular and bimolecular massaction reactions are allowed
assign_products -- function to decide where products are assigned
get_rate -- function to get the rate of a reaction between neighboring nodes

The ordering of species is:
[species in node 1, species in node 2, ... ] <-- lengths = num_species, num_species, ...

The ordering of reactions is:
[instances of reaction 1, instances of reaction 2, ..., <-- lengths = num_neighboring_pairs, num_neighboring_pairs, ...
diffusions of species 1, diffusions of species 2, ...   <-- lengths = sum_degrees, sum_degrees, ...]

diffusions of species i = diffusions from node 1, diffusions from node 2, ... <-- lengths = degree of node 1, degree of node 2, ...
"""
function to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob, alg, assign_products, get_rate)
    num_nodes = length(connectivity_list)
    sum_degrees = (length ∘ vec ∘ hcat)(connectivity_list...)
    massaction_jump = jump_prob.massaction_jump
    @unpack reactant_stoch, net_stoch = massaction_jump
    num_majumps = get_num_majumps(massaction_jump)
    num_species = length(jump_prob.prob.u0)
    prob = jump_prob.prob
    bimolecular_rxs_with_same_reactants = [rx for rx in 1:num_majumps if is_bimolecular_with_same_reactants(reactant_stoch[rx])]
    bimolecular_rxs_with_different_reactants = [rx for rx in 1:num_majumps if is_bimolecular_with_different_reactants(reactant_stoch[rx])]
    num_bimolecular_rxs_with_same_reactants = length(bimolecular_rxs_with_same_reactants)
    num_bimolecular_rxs_with_different_reactants = length(bimolecular_rxs_with_different_reactants)
    rxs_reactants = [Tuple([s for (s,c) in reactant_stoch[rx]]) for rx in 1:num_majumps]
    full_net_stoichiometries = [get_full_net_stoichiometry(rx, reactant_stoch, net_stoch) for rx in 1:num_majumps]

    # spatial constants
    neighboring_pairs = get_neiboring_pairs(connectivity_list)
    num_spacial_majumps = num_nodes * num_majumps + length(neighboring_pairs) * (num_bimolecular_rxs_with_same_reactants + 2*num_bimolecular_rxs_with_different_reactants)
    num_spatial_species = num_species*num_nodes
    source_target_index_pairs = get_source_target_index_pairs(connectivity_list)
    num_spatial_rxs = num_spacial_majumps + num_species*sum_degrees

    # preallocate stoichiometry and rates arrays
    rx_rates = Array{Float64,1}(undef, num_spatial_rxs)
    reaction_stoichiometries = Array{Array{Pair{Int64,Int64},1},1}(undef, num_spatial_rxs)
    net_stoichiometries = Array{Array{Pair{Int64,Int64},1},1}(undef, num_spatial_rxs)

    # make stoichiometries for reactions within nodes
    for (i, (node, rx)) in enumerate(Iterators.product(1:num_nodes, 1:num_majumps))
        rx_rates[i] = prob.p[rx]
        reaction_stoichiometries[i], net_stoichiometries[i] = get_rx_stoichiometries(node, rx, massaction_jump, num_species)
    end

    # make stoichiometries for bimolecular_rxs_with_same_reactants
    for (i, ((node1, node2), rx)) in enumerate(Iterators.product(neighboring_pairs, bimolecular_rxs_with_same_reactants))
        ind = i + num_nodes * num_majumps
        rx_rates[ind] = get_rate(rx, (node1, source_species), (node2, target_species), prob.p)
        reactant1, reactant2 = rxs_reactants[rx]
        products1, products2 = assign_products(rx, massaction_jump, (node1, reactant1), (node2, reactant2))
        reaction_stoichiometries[ind], net_stoichiometries[ind] = get_bimolecular_rx_stoichiometries((node1, reactant1, products1), (node2, reactant2, products2), rx, massaction_jump, num_species)
    end

    # make stoichiometries for bimolecular_rxs_with_different_reactants
    for (i, (j,(node1, node2), rx)) in enumerate(Iterators.product(0:1, neighboring_pairs, bimolecular_rxs_with_different_reactants))
        ind = i + num_nodes * num_majumps + length(neighboring_pairs) * num_bimolecular_rxs_with_same_reactants
        if j == 0 # species 1 in reactants is in node 1, and species 2 in reactants is in node 2
            reactant1, reactant2 = rxs_reactants[rx]
        else # species 1 in reactants is in node 2, and species 2 in reactants is in node 1
            reactant1, reactant2 = reverse(rxs_reactants[rx])
        end
        products1, products2 = assign_products(rx, massaction_jump, (node1, reactant1), (node2, reactant2))
        reaction_stoichiometries[ind], net_stoichiometries[ind] = get_bimolecular_rx_stoichiometries((node1, reactant1, products1), (node2, reactant2, products2), rx, massaction_jump, num_species)
        rx_rates[ind] = get_rate(rx, (node1, reactant1), (node2, reactant2), prob.p)
    end

    # make stoichiometries for diffusions
    for (i, ((source, target_index), species)) in enumerate(Iterators.product(source_target_index_pairs, 1:num_species))
        rx_rates[i+num_spacial_majumps] = diff_rates[source][target_index][species]
        reaction_stoichiometries[i+num_spacial_majumps], net_stoichiometries[i+num_spacial_majumps] = get_diff_stoichiometries(source, connectivity_list[source][target_index], species, connectivity_list, num_species)
    end

    spatial_majumps = MassActionJump(rx_rates, reaction_stoichiometries, net_stoichiometries)

    # NOTE: arbitrary decision to copy the original u0
    starting_state = vec(hcat([prob.u0 for i in 1:num_nodes]...))
    spatial_prob = DiscreteProblem(starting_state, prob.tspan, rx_rates)
    spec_to_dep_rxs = DiffEqJump.spec_to_dep_rxs_map(num_spatial_species, spatial_majumps)
    rxs_to_dep_spec = rxs_to_dep_spec_map(spatial_majumps)
    JumpProblem(spatial_prob, alg, spatial_majumps, save_positions=(false,false), vartojumps_map=spec_to_dep_rxs, jumptovars_map=rxs_to_dep_spec)
end

############ Helper functions ###############
"get the stoichiometries for bimolecular reaction rx (non-spatial index) between node1 and node2."
function get_bimolecular_rx_stoichiometries((node1, reactant1, products1), (node2, reactant2, products2), rx, massaction_jump, num_species)
    @assert is_bimolecular(massaction_jump.reactant_stoch[rx])
    @assert vcat(products1, products2) == [s => c for (s,c) in get_full_net_stoichiometry(rx, massaction_jump.reactant_stoch, massaction_jump.net_stoch) if c > 0]
    [to_spatial_spec(node1, reactant1, num_species) => 1, to_spatial_spec(node2, reactant2, num_species) => 1],
    vcat([to_spatial_spec(node1, reactant1, num_species) => -1], [to_spatial_spec(node2, reactant2, num_species) => -1], [to_spatial_spec(node1, product, num_species) => coeff for (product, coeff) in products1], [to_spatial_spec(node2, product, num_species) => coeff for (product, coeff) in products2])
end

"make a map from reactions to dependent species"
function rxs_to_dep_spec_map(majumps)
    [[s for (s, c) in majumps.net_stoch[i]] for i in 1:get_num_majumps(majumps)]
end
"get all bimolecular reactions, along with a boolean indicating if the two reactants are of the same species. The boolean is true iff the two reactants are of the same species"
function get_bimolecular_reactions(reactstoch)
    [rx => length(reactstoch[rx]) == 1 for rx in 1:length(reactstoch) if is_bimolecular(reactstoch[rx])]
end

"return true iff the reaction with rx_stoichiometry is bimolecular with reactants of same species"
function is_bimolecular_with_same_reactants(rx_stoichiometry)
    length(rx_stoichiometry) == 1 && rx_stoichiometry[1][2] == 2
end
"return true iff the reaction with rx_stoichiometry is bimolecular with reactants of different species"
function is_bimolecular_with_different_reactants(rx_stoichiometry)
    length(rx_stoichiometry) == 2 && rx_stoichiometry[1][2] == 1 && rx_stoichiometry[2][2] == 1
end
"true iff the reaction with this stoichiometry is bimolecular"
function is_bimolecular(rx_stoichiometry)
    is_bimolecular_with_same_reactants(rx_stoichiometry) || is_bimolecular_with_different_reactants(rx_stoichiometry)
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
function get_source_target_index_pairs(connectivity_list)
    vec(hcat([[Pair(source, target_index) for target_index in 1:length(connectivity_list[source])] for source in 1:length(connectivity_list)]...))
end

"get pairs of neighboring nodes. If the graph is undirected, equivalent to the list of edges"
function get_neiboring_pairs(connectivity_list)
    source_target_pairs = [let target = connectivity_list[source][target_index]
                                target < source ? target => source : source => target
                            end for (source,target_index) in get_source_target_index_pairs(connectivity_list)]
    unique(source_target_pairs)
end

"get the full net stoichiometry vector of massaction reaction rx, with all reactants and products, not only those whose number changes"
function get_full_net_stoichiometry(rx, reactstoch, netstoch)
    full_net_stoichiometry = deepcopy(netstoch[rx])
    for (s, c) in reactstoch[rx]
        index = findfirst(x-> x[1] == s, full_net_stoichiometry)
        if index == nothing
            push!(full_net_stoichiometry, s => -c)
            push!(full_net_stoichiometry, s => c)
        elseif full_net_stoichiometry[index][2] != -c
            push!(full_net_stoichiometry, s => full_net_stoichiometry[index][2] + c)
            full_net_stoichiometry[index] = s => -c
        end
    end
    full_net_stoichiometry
end
