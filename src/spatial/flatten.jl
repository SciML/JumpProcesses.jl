using DiffEqJump, DiffEqBase, LightGraphs

rates = [0.1, 1.0]
netstoch = [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]]
reactstoch = [[1 => 1, 2 => 1], [3 => 1]]

graph = grid((2,2))
num_species = 3
num_nodes = nv(graph)
end_time = 1.0
tspan = (0.0, end_time)
u0 = ones(Int, num_species, num_nodes)
alg = Direct()

hopping_rate = 1.0
hopping_constants = Vector{Matrix{Float64}}(undef, num_nodes) # TODO QUESTION is this how we want to accept hopping constants? 
for site in 1:num_nodes
    hopping_constants[site] = hopping_rate*ones(num_species, DiffEqJump.num_neighbors(graph, site))
end

#TODO QUESTION should accept variable rx rates?

# reaction
function flatten(netstoch::AbstractArray, reactstoch::AbstractArray, rates::AbstractArray, graph::AbstractGraph, u0::Matrix{Int}, tspan, alg, hopping_constants::Vector{Matrix{F}}; kwargs...) where F <: Number
    graph = SimpleDiGraph(graph)
    num_rxs = length(rates)
    num_species = size(u0, 1)
    num_nodes = nv(graph)
    @assert size(u0, 2) == num_nodes
    spec_CI = CartesianIndices((num_species, num_nodes))
    spec_LI = LinearIndices((num_species, num_nodes))

    #reactions
    function to_spatial(stoch, num_nodes)
        out = []
        for site in 1:num_nodes
            push!(out, map(rx_stoch -> map(p -> Pair(spec_LI[spec_CI[p[1],site]], p[2]), rx_stoch), stoch))
        end
        vcat(out...)
    end
    rx_netstoch =  to_spatial(netstoch, num_nodes)
    rx_reactstoch =  to_spatial(reactstoch, num_nodes)
    rx_rates = vcat([rates for i in 1:num_nodes]...)

    #hops
    hop_netstoch = []
    hop_reacstoch = []
    hop_rates = []

    # assuming hopping_constants isa Vector{Matrix} where hopping_constants[src][species, index] is the hop const of species from src to neighbor at index.
    for src in vertices(graph)
        hopping_constants_at_src = hopping_constants[src]
        for (i,dst) in enumerate(neighbors(graph, src))
            for species in 1:num_species
                dst_idx = spec_LI[spec_CI[species,dst]]
                src_idx = spec_LI[spec_CI[species,src]]
                push!(hop_netstoch, [dst_idx => 1, src_idx => -1])
                push!(hop_reacstoch, [src_idx => 1])
                push!(hop_rates, hopping_constants_at_src[species, i])
            end
        end
    end

    total_netstoch = convert(Vector{Vector{Pair{Int, Int}}}, vcat(hop_netstoch, rx_netstoch))
    total_reacstoch = convert(Vector{Vector{Pair{Int, Int}}}, vcat(hop_reacstoch, rx_reactstoch))
    total_rates = convert(Vector{Float64}, vcat(hop_rates, rx_rates))
    ma_jump = MassActionJump(total_rates, total_reacstoch, total_netstoch)

    flattened_u0 = reshape(u0, length(u0))
    prob = DiscreteProblem(flattened_u0, tspan, total_rates)

    JumpProblem(prob, alg, ma_jump; kwargs...)
end