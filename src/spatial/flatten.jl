using DiffEqJump, DiffEqBase, LightGraphs

"""
prob.u0 must be a Matrix with prob.u0[i,j] being the number of species i at site j
prob.p must be unscaled rates, either an array or a matrix with prob.p[i,j] being the rate of reaction j at site j
"""
function flatten(ma_jump::MassActionJump, prob::DiscreteProblem, spatial_system, hopping_constants; kwargs...)
    netstoch = ma_jump.net_stoch
    reactstoch = ma_jump.reactant_stoch
    rx_rates = ma_jump.scaled_rates
    tspan = prob.tspan
    u0 = prob.u0
    flatten(netstoch, reactstoch, rx_rates, spatial_system, u0, tspan, hopping_constants; scale_rates = false, kwargs...)
end

"""
if hopping_constants is a matrix, assume hopping_constants[i,j] is species i, site j
"""
function flatten(netstoch::AbstractArray, reactstoch::AbstractArray, rx_rates::Vector, spatial_system, u0::Matrix{Int}, tspan, hopping_constants::Matrix{F}; kwargs...) where F <: Number
    num_nodes = num_sites(spatial_system)
    num_specs = size(u0, 1)
    @assert size(hopping_constants) == size(u0) # hopping_constants[i,j] is species i, site j
    hop_constants = Vector{F}(undef, size(hopping_constants, 2))
    for site in 1:num_nodes
        num_nbs = num_neighbors(spatial_system, site)
        hop_constants[site] = reshape(repeat(hopping_constants[:,site], num_nbs), num_specs, num_nodes)
    end
    
    flatten(netstoch, reactstoch, rates, spatial_system, u0, tspan, hop_constants; kwargs...)
end

"""
if reaction rates is a vector, assume reaction rates are equal across sites
"""
function flatten(netstoch::AbstractArray, reactstoch::AbstractArray, rx_rates::Vector, spatial_system, u0::Matrix{Int}, tspan, hopping_constants::Vector{Matrix{F}}; kwargs...) where F <: Number
    num_nodes = num_sites(spatial_system)
    rates = reshape(repeat(rx_rates, num_nodes), length(rx_rates), num_nodes)
    flatten(netstoch, reactstoch, rates, spatial_system, u0, tspan, hopping_constants; kwargs...)
end

"""
convert to Float64 if type Any
"""
function flatten(netstoch::AbstractArray{Any}, reactstoch::AbstractArray{Any}, rx_rates::Vector{Any}, spatial_system, u0::Matrix{Int}, tspan, hopping_constants::Vector{Matrix{F}}; kwargs...) where F <: Number
    net_stoch = convert(Vector{Vector{Pair{Int64, Int64}}}, netstoch)
    react_stoch = convert(Vector{Vector{Pair{Int64, Int64}}}, reactstoch)
    rates = convert(Vector{Float64}, rx_rates)
    flatten(net_stoch, react_stoch, rates, spatial_system, u0, tspan, hopping_constants; kwargs...)
end

"""
"flatten" the spatial jump problem. Return flattened DiscreteProblem and MassActionJump.
"""
function flatten(netstoch::Vector{R}, reactstoch::Vector{R}, rx_rates::Matrix{F}, spatial_system, u0::Matrix{Int}, tspan, hopping_constants::Vector{Matrix{F}}; scale_rates = true, kwargs...) where {R, F <: Number}
    num_species = size(u0, 1)
    num_nodes = num_sites(spatial_system)
    num_rxs = length(reactstoch)
    @assert length(hopping_constants) == size(u0, 2) == size(rx_rates, 2) == num_nodes
    @assert length(netstoch) == length(reactstoch) == size(rx_rates, 1) == num_rxs
    spec_CI = CartesianIndices((num_species, num_nodes))
    spec_LI = LinearIndices((num_species, num_nodes))

    sum_outdegrees = sum(num_neighbors(spatial_system, site) for site in 1:num_nodes)
    num_jumps = num_species*(sum_outdegrees + num_rxs)
    total_netstoch = R[]; sizehint!(total_netstoch, num_jumps)
    total_reactstoch = R[]; sizehint!(total_reactstoch, num_jumps)
    total_rates = F[]; sizehint!(total_rates, num_jumps)

    #hops
    # assuming hopping_constants isa Vector{Matrix} where hopping_constants[src][species, index] is the hop const of species from src to neighbor at index in neighbors(spatial_system, src)
    for src in 1:num_nodes
        hopping_constants_at_src = hopping_constants[src]
        for (i,dst) in enumerate(neighbors(spatial_system, src))
            for species in 1:num_species
                dst_idx = spec_LI[spec_CI[species,dst]]
                src_idx = spec_LI[spec_CI[species,src]]
                push!(total_netstoch, [dst_idx => 1, src_idx => -1])
                push!(total_reactstoch, [src_idx => 1])
                push!(total_rates, hopping_constants_at_src[species, i])
            end
        end
    end

    #reactions
    for site in 1:num_nodes
        append!(total_netstoch, map(rx_stoch -> map(p -> Pair(spec_LI[spec_CI[p[1],site]], p[2]), rx_stoch), netstoch))
        append!(total_reactstoch, map(rx_stoch -> map(p -> Pair(spec_LI[spec_CI[p[1],site]], p[2]), rx_stoch), reactstoch))
    end
    append!(total_rates, vec(rx_rates)) # assuming rx_rates isa Matrix where rx_rates[rx, site] is the rate of rx at site

    # put everything together
    ma_jump = MassActionJump(total_rates, total_reactstoch, total_netstoch; nocopy = true, scale_rates = scale_rates)
    flattened_u0 = vec(u0)
    prob = DiscreteProblem(flattened_u0, tspan, total_rates)
    prob, ma_jump
end