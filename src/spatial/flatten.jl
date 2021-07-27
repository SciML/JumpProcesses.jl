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
if reaction rates is a vector, assume reaction rates are equal across sites
"""
function flatten(netstoch::AbstractArray, reactstoch::AbstractArray, rx_rates::Vector, spatial_system, u0::Matrix{Int}, tspan, hopping_constants::Vector{Matrix{F}}; kwargs...) where F <: Number
    num_nodes = num_sites(spatial_system)
    rates = reshape(repeat(rx_rates, num_nodes), length(rx_rates), num_nodes)
    flatten(netstoch, reactstoch, rates, spatial_system, u0, tspan, hopping_constants; kwargs...)
end

"""
"flatten" the spatial jump problem. Return flattened DiscreteProblem and MassActionJump.
"""
function flatten(netstoch::AbstractArray, reactstoch::AbstractArray, rx_rates::Matrix, spatial_system, u0::Matrix{Int}, tspan, hopping_constants::Vector{Matrix{F}}; scale_rates = true, kwargs...) where F <: Number
    num_species = size(u0, 1)
    num_nodes = num_sites(spatial_system)
    @assert size(u0, 2) == num_nodes
    @assert length(hopping_constants) == num_nodes
    spec_CI = CartesianIndices((num_species, num_nodes))
    spec_LI = LinearIndices((num_species, num_nodes))

    #reactions
    function to_spatial(stoch, num_nodes, spec_LI, spec_CI)
        out = eltype(stoch)[]
        sizehint!(out, length(stoch)*num_nodes)
        for site in 1:num_nodes
            append!(out, map(rx_stoch -> map(p -> Pair(spec_LI[spec_CI[p[1],site]], p[2]), rx_stoch), stoch))
        end
        out
    end
    rx_netstoch =  to_spatial(netstoch, num_nodes, spec_LI, spec_CI)
    rx_reactstoch =  to_spatial(reactstoch, num_nodes, spec_LI, spec_CI)

    #hops
    hop_netstoch = eltype(netstoch)[]; sizehint!(hop_netstoch, num_nodes*num_species*6)
    hop_reacstoch = eltype(netstoch)[]; sizehint!(hop_reacstoch, num_nodes*num_species*6)
    hop_rates = F[]; sizehint!(hop_rates, num_nodes*num_species*6)

    # assuming hopping_constants isa Vector{Matrix} where hopping_constants[src][species, index] is the hop const of species from src to neighbor at index.
    for src in 1:num_nodes
        hopping_constants_at_src = hopping_constants[src]
        for (i,dst) in enumerate(neighbors(spatial_system, src))
            for species in 1:num_species
                dst_idx = spec_LI[spec_CI[species,dst]]
                src_idx = spec_LI[spec_CI[species,src]]
                push!(hop_netstoch, [dst_idx => 1, src_idx => -1])
                push!(hop_reacstoch, [src_idx => 1])
                push!(hop_rates, hopping_constants_at_src[species, i])
            end
        end
    end

    total_netstoch = vcat(hop_netstoch, rx_netstoch)
    total_reacstoch = vcat(hop_reacstoch, rx_reactstoch)
    total_rates = vcat(hop_rates, vec(rx_rates))
    ma_jump = MassActionJump(total_rates, total_reacstoch, total_netstoch; scale_rates = scale_rates)

    flattened_u0 = vec(u0)
    prob = DiscreteProblem(flattened_u0, tspan, total_rates)

    prob, ma_jump
end