using DiffEqJump, DiffEqBase, Parameters

# NOTE: save_positions = (false, false) for DiscreteProblem by default
"""
Construct a jump problem, where diffusions are represented as reactions.
Given:
adjacency/connectivity list -- representing a graph. Can be directed.
massaction_jump -- if neighbors_react is true, only uni- and bi-molecular reactions are allowed
prob -- DiscreteProblem
alg -- algorithm to use
(optional keyword arg) diff_rates -- diffusion rates (analagous to reaction rates). 0.0 by default.
(optional keyword arg) save_positions -- when/whether to save positions. Equal to (false, false) by default.
(optional keyword arg) starting_state -- note the ordering of species. Fills all nodes with prob.u0 by default.
(optional keyword arg) get_rate -- function to get the rate of a reaction between neighboring nodes. Equal to (rx, _, _, rates) -> rates[rx] by defualt
(optional keyword arg) assign_products -- function to decide where products are assigned

The ordering of species is:
[species in node 1, species in node 2, ... ] <-- lengths = num_species, num_species, ...
"""
function JumpProblem(prob, aggregator::WellMixedSpatial, massaction_jump::MassActionJump, connectivity_list; diff_rates = 0.0, save_positions = typeof(prob) <: DiffEqBase.AbstractDiscreteProblem ? (false,false) : (true,true), starting_state = vcat([prob.u0 for i in 1:length(connectivity_list)]...), get_rate = (rx, _, _, rates) -> rates[rx], assign_products = nothing, kwargs...)
    to_spatial_jump_prob(connectivity_list, massaction_jump, prob, aggregator.WellMixedSSA;  diff_rates = diff_rates, save_positions = save_positions, starting_state = starting_state, get_rate = get_rate, assign_products = assign_products, kwargs...)
end


"struct to hold the spatial constants"
struct Spatial_Constants
    num_nodes
    sum_degrees
    reactant_stoch
    net_stoch
    num_majumps
    num_species
    bimolecular_rxs_with_same_reactants
    bimolecular_rxs_with_different_reactants
    num_bimolecular_rxs_with_same_reactants
    num_bimolecular_rxs_with_different_reactants
    rxs_reactants
    full_net_stoichiometries
    neighboring_pairs
    num_spacial_majumps
    num_spatial_species
    source_target_index_pairs
    num_spatial_rxs
    connectivity_list
    massaction_jump
    prob
    diff_rates
    starting_state
    neighbors_react
end

"""
Construct a jump problem, where diffusions are represented as reactions.
Given:
adjacency/connectivity list -- representing a graph. Can be directed.
massaction_jump -- if neighbors_react is true, only uni- and bi-molecular reactions are allowed
prob -- DiscreteProblem
alg -- algorithm to use
(optional keyword arg) diff_rates -- diffusion rates (analagous to reaction rates). 0.0 by default.
(optional keyword arg) save_positions -- when/whether to save positions. Equal to (false, false) by default.
(optional keyword arg) starting_state -- note the ordering of species. Fills all nodes with prob.u0 by default.
(optional keyword arg) get_rate -- function to get the rate of a reaction between neighboring nodes. Equal to (rx, _, _, rates) -> rates[rx] by defualt
(optional keyword arg) assign_products -- function to decide where products are assigned

The ordering of species is:
[species in node 1, species in node 2, ... ] <-- lengths = num_species, num_species, ...
"""
function to_spatial_jump_prob(connectivity_list, massaction_jump :: MassActionJump, prob :: DiscreteProblem, alg;  diff_rates = 0.0, save_positions = (false, false), starting_state = vcat([prob.u0 for i in 1:length(connectivity_list)]...), get_rate = (rx, _, _, rates) -> rates[rx], assign_products = nothing, kwargs...)

    if diff_rates isa Number
        diff_rates_for_edge = ones(length(prob.u0))*diff_rates
        diff_rates = [[diff_rates_for_edge for j in 1:length(connectivity_list[i])] for i in 1:length(connectivity_list)]
    elseif diff_rates isa Array
        @assert length(diff_rates) == length(prob.u0)
        diff_rates = [[diff_rates*1.0 for j in 1:length(connectivity_list[i])] for i in 1:length(connectivity_list)]
    end

    rx_rates, spatial_majumps = get_spatial_rates_and_massaction_jumps(connectivity_list, diff_rates, massaction_jump, prob, alg; save_positions = save_positions, starting_state = starting_state, get_rate = get_rate, assign_products = assign_products)

    num_spatial_species = length(connectivity_list)*length(prob.u0)
    spatial_prob = DiscreteProblem(starting_state, prob.tspan, rx_rates)
    spec_to_dep_rxs = DiffEqJump.spec_to_dep_rxs_map(num_spatial_species, spatial_majumps)
    rxs_to_dep_spec = DiffEqJump.rxs_to_dep_spec_map(spatial_majumps)
    JumpProblem(spatial_prob, alg, spatial_majumps, save_positions = save_positions, vartojumps_map=spec_to_dep_rxs, jumptovars_map=rxs_to_dep_spec, kwargs...)
end

"""
Construct an array of (unscaled) reaction rates and a MassActionJump object.
The ordering of reactions is:
[instances of reaction 1, instances of reaction 2, ..., <-- lengths = num_nodes/num_neighboring_pairs, num_nodes/num_neighboring_pairs, ...
diffusions of species 1, diffusions of species 2, ...   <-- lengths = sum_degrees, sum_degrees, ...]

diffusions of species i = diffusions from node 1, diffusions from node 2, ... <-- lengths = degree of node 1, degree of node 2, ...
"""
function get_spatial_rates_and_massaction_jumps(connectivity_list, diff_rates, massaction_jump :: MassActionJump, prob :: DiscreteProblem, alg; save_positions = (false, false), starting_state = nothing, get_rate = (rx, _, _, rates) -> rates[rx], assign_products = nothing)

    neighbors_react = (assign_products != nothing)

    spatial_constants = Spatial_Constants(connectivity_list, diff_rates, starting_state, massaction_jump :: MassActionJump, prob :: DiscreteProblem, false)

    if neighbors_react
        rx_rates, spatial_majumps = get_spatial_majumps(spatial_constants, get_rate, assign_products)
    else
        rx_rates, spatial_majumps = get_spatial_majumps(spatial_constants)
    end
    rx_rates, spatial_majumps
end

"""
Construct spatial massaction jumps.
The ordering of reactions is:
[instances of reaction 1, instances of reaction 2, ..., <-- lengths = num_nodes/num_neighboring_pairs, num_nodes/num_neighboring_pairs, ...
diffusions of species 1, diffusions of species 2, ...   <-- lengths = sum_degrees, sum_degrees, ...]

diffusions of species i = diffusions from node 1, diffusions from node 2, ... <-- lengths = degree of node 1, degree of node 2, ...
"""
function get_spatial_majumps(spatial_constants :: Spatial_Constants, get_rate = nothing, assign_products = nothing)
    @unpack num_spatial_rxs = spatial_constants
    # preallocate stoichiometry and rates arrays
    rx_rates = Array{Float64,1}(undef, num_spatial_rxs)
    reaction_stoichiometries = Array{Array{Pair{Int64,Int64},1},1}(undef, num_spatial_rxs)
    net_stoichiometries = Array{Array{Pair{Int64,Int64},1},1}(undef, num_spatial_rxs)

    # rates and stoichiometries for within-node reactions and diffusions
    fill_rates_and_stoichiometries!(rx_rates, reaction_stoichiometries, net_stoichiometries, spatial_constants)

    if get_rate != nothing && assign_products != nothing
        # rates and stoichiometries for across-nodes bimolecular reactions
        fill_rates_and_stoichiometries_neighbors_reacting!(rx_rates, reaction_stoichiometries, net_stoichiometries, spatial_constants, get_rate, assign_products)
    end

    # remove reactions with zero rates
    zero_rate_indices = []
    for (i, r) in enumerate(rx_rates)
        r < eps(r) && push!(zero_rate_indices, i)
    end
    deleteat!(rx_rates, zero_rate_indices)
    deleteat!(reaction_stoichiometries, zero_rate_indices)
    deleteat!(net_stoichiometries, zero_rate_indices)

    rx_rates, MassActionJump(rx_rates, reaction_stoichiometries, net_stoichiometries)
end

############ Helper functions ###############
"get the stoichiometries for bimolecular reaction rx (non-spatial index) between node1 and node2."
function get_bimolecular_rx_stoichiometries((node1, reactant1, products1), (node2, reactant2, products2), rx, massaction_jump, num_species)
    @assert is_bimolecular(massaction_jump.reactant_stoch[rx])
    @assert vcat(products1, products2) == [s => c for (s,c) in get_full_net_stoichiometry(rx, massaction_jump.reactant_stoch, massaction_jump.net_stoch) if c > 0]
    [to_spatial_spec(node1, reactant1, num_species) => 1, to_spatial_spec(node2, reactant2, num_species) => 1],
    vcat([to_spatial_spec(node1, reactant1, num_species) => -1], [to_spatial_spec(node2, reactant2, num_species) => -1], [to_spatial_spec(node1, product, num_species) => coeff for (product, coeff) in products1], [to_spatial_spec(node2, product, num_species) => coeff for (product, coeff) in products2])
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
    vcat([[Pair(source, target_index) for target_index in 1:length(connectivity_list[source])] for source in 1:length(connectivity_list)]...)
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
        if index === nothing
            push!(full_net_stoichiometry, s => -c)
            push!(full_net_stoichiometry, s => c)
        elseif full_net_stoichiometry[index][2] != -c
            push!(full_net_stoichiometry, s => full_net_stoichiometry[index][2] + c)
            full_net_stoichiometry[index] = s => -c
        end
    end
    full_net_stoichiometry
end

"Fill rates and stoichiometries with within-node reactions and diffusions"
function fill_rates_and_stoichiometries!(rx_rates, reaction_stoichiometries, net_stoichiometries, spatial_constants :: Spatial_Constants)
    @unpack num_nodes, num_majumps, num_species, num_spacial_majumps, prob, massaction_jump, source_target_index_pairs, diff_rates, connectivity_list = spatial_constants
    # fill stoichiometries for reactions
    for (i, (node, rx)) in enumerate(Iterators.product(1:num_nodes, 1:num_majumps))
        rx_rates[i] = prob.p[rx]
        reaction_stoichiometries[i], net_stoichiometries[i] = get_rx_stoichiometries(node, rx, massaction_jump, num_species)
    end
    # fill stoichiometries for diffusions
    for (i, ((source, target_index), species)) in enumerate(Iterators.product(source_target_index_pairs, 1:num_species))
        rx_rates[i+num_spacial_majumps] = diff_rates[source][target_index][species]
        reaction_stoichiometries[i+num_spacial_majumps], net_stoichiometries[i+num_spacial_majumps] = get_diff_stoichiometries(source, connectivity_list[source][target_index], species, connectivity_list, num_species)
    end
    nothing
end

"Fill rates and stoichiometries with across-nodes bimolecular reactions"
function fill_rates_and_stoichiometries_neighbors_reacting!(rx_rates, reaction_stoichiometries, net_stoichiometries, spatial_constants :: Spatial_Constants, get_rate, assign_products)
    @unpack neighboring_pairs, bimolecular_rxs_with_same_reactants, bimolecular_rxs_with_different_reactants, num_majumps, num_nodes, num_bimolecular_rxs_with_same_reactants, prob, rxs_reactants, massaction_jump, num_species = spatial_constants
    # make stoichiometries for bimolecular_rxs_with_same_reactants
    for (i, ((node1, node2), rx)) in enumerate(Iterators.product(neighboring_pairs, bimolecular_rxs_with_same_reactants))
        ind = i + num_nodes * num_majumps
        reactant1, reactant2 = rxs_reactants[rx]
        @assert reactant1 == reactant2

        rx_rates[ind] = get_rate(rx, (node1, reactant1), (node2, reactant2), prob.p)
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
        @assert reactant1 != reactant2

        rx_rates[ind] = get_rate(rx, (node1, reactant1), (node2, reactant2), prob.p)
        products1, products2 = assign_products(rx, massaction_jump, (node1, reactant1), (node2, reactant2))
        reaction_stoichiometries[ind], net_stoichiometries[ind] = get_bimolecular_rx_stoichiometries((node1, reactant1, products1), (node2, reactant2, products2), rx, massaction_jump, num_species)
    end
    nothing
end

"initialize constants"
function Spatial_Constants(connectivity_list, diff_rates, starting_state, massaction_jump :: MassActionJump, prob :: DiscreteProblem, neighbors_react)
    num_nodes = length(connectivity_list)
    sum_degrees = sum([length(nbs) for nbs in connectivity_list])
    @unpack reactant_stoch, net_stoch = massaction_jump
    num_majumps = get_num_majumps(massaction_jump)
    num_species = length(prob.u0)

    bimolecular_rxs_with_same_reactants = [rx for rx in 1:num_majumps if is_bimolecular_with_same_reactants(reactant_stoch[rx])]
    bimolecular_rxs_with_different_reactants = [rx for rx in 1:num_majumps if is_bimolecular_with_different_reactants(reactant_stoch[rx])]
    num_bimolecular_rxs_with_same_reactants = length(bimolecular_rxs_with_same_reactants)
    num_bimolecular_rxs_with_different_reactants = length(bimolecular_rxs_with_different_reactants)
    rxs_reactants = [Tuple([s for (s,c) in reactant_stoch[rx]]) for rx in 1:num_majumps]
    full_net_stoichiometries = [get_full_net_stoichiometry(rx, reactant_stoch, net_stoch) for rx in 1:num_majumps]

    # spatial constants
    neighboring_pairs = get_neiboring_pairs(connectivity_list)
    if neighbors_react
        num_spacial_majumps = num_nodes * num_majumps + length(neighboring_pairs) * (num_bimolecular_rxs_with_same_reactants + 2*num_bimolecular_rxs_with_different_reactants)
    else
        num_spacial_majumps = num_nodes * num_majumps
    end
    num_spatial_species = num_species*num_nodes
    source_target_index_pairs = get_source_target_index_pairs(connectivity_list)
    num_spatial_rxs = num_spacial_majumps + num_species*sum_degrees
    Spatial_Constants(num_nodes, sum_degrees, reactant_stoch, net_stoch, num_majumps, num_species, bimolecular_rxs_with_same_reactants, bimolecular_rxs_with_different_reactants, num_bimolecular_rxs_with_same_reactants, num_bimolecular_rxs_with_different_reactants, rxs_reactants, full_net_stoichiometries, neighboring_pairs, num_spacial_majumps, num_spatial_species, source_target_index_pairs, num_spatial_rxs, connectivity_list, massaction_jump, prob, diff_rates, starting_state, neighbors_react)
end
