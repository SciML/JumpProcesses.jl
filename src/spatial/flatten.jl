using JumpProcesses, DiffEqBase, Graphs

"""
prob.u0 must be a Matrix with prob.u0[i,j] being the number of species i at site j
"""
function flatten(
        ma_jump, prob::DiscreteProblem, spatial_system, hopping_constants;
        kwargs...
    )
    tspan = prob.tspan
    u0 = prob.u0
    if ma_jump === nothing
        ma_jump = MassActionJump(
            Vector{typeof(tspan[1])}(),
            Vector{Vector{Pair{Int, eltype(u0)}}}(),
            Vector{Vector{Pair{Int, eltype(u0)}}}()
        )
    end
    netstoch = ma_jump.net_stoch
    reactstoch = ma_jump.reactant_stoch
    rx_rates = if isa(ma_jump, MassActionJump)
        ma_jump.scaled_rates
    elseif isa(ma_jump, SpatialMassActionJump)
        num_nodes = num_sites(spatial_system)
        if isnothing(ma_jump.uniform_rates) && isnothing(ma_jump.spatial_rates)
            zeros(0, num_nodes)
        elseif isnothing(ma_jump.uniform_rates)
            ma_jump.spatial_rates
        elseif isnothing(ma_jump.spatial_rates)
            reshape(
                repeat(ma_jump.uniform_rates, num_nodes),
                length(ma_jump.uniform_rates), num_nodes
            )
        else
            @assert size(ma_jump.spatial_rates, 2) == num_nodes
            cat(
                dims = 1,
                reshape(
                    repeat(ma_jump.uniform_rates, num_nodes),
                    length(ma_jump.uniform_rates), num_nodes
                ),
                ma_jump.spatial_rates
            )
        end
    else
        error("flatten: unsupported jump type $(typeof(ma_jump))")
    end
    return flatten(
        netstoch, reactstoch, rx_rates, spatial_system, u0, tspan, hopping_constants;
        scale_rates = false, kwargs...
    )
end

"""
if hopping_constants is a matrix, assume hopping_constants[i,j] is the hopping constant of species i from site j to any neighbor
"""
function flatten(
        netstoch::AbstractArray, reactstoch::AbstractArray,
        rx_rates::AbstractArray, spatial_system, u0::Matrix{Int}, tspan,
        hopping_constants::Matrix{F}; kwargs...
    ) where {F <: Number}
    @assert size(hopping_constants) == size(u0)
    hop_constants = Matrix{Vector{F}}(undef, size(hopping_constants))
    for ci in CartesianIndices(hop_constants)
        (species, site) = Tuple(ci)
        hop_constants[ci] = hopping_constants[species, site] * ones(outdegree(spatial_system, site))
    end
    return flatten(
        netstoch, reactstoch, rx_rates, spatial_system, u0, tspan, hop_constants;
        kwargs...
    )
end

"""
if reaction rates is a vector, assume reaction rates are equal across sites
"""
function flatten(
        netstoch::AbstractArray, reactstoch::AbstractArray, rx_rates::Vector,
        spatial_system, u0::Matrix{Int}, tspan,
        hopping_constants::Matrix{Vector{F}}; kwargs...
    ) where {F <: Number}
    num_nodes = num_sites(spatial_system)
    rates = reshape(repeat(rx_rates, num_nodes), length(rx_rates), num_nodes)
    return flatten(
        netstoch, reactstoch, rates, spatial_system, u0, tspan, hopping_constants;
        kwargs...
    )
end

"""
"flatten" the spatial jump problem. Return flattened DiscreteProblem and MassActionJump.
"""
function flatten(
        netstoch::Vector{R}, reactstoch::Vector{R}, rx_rates::Matrix{F},
        spatial_system, u0::Matrix{Int}, tspan,
        hopping_constants::Matrix{Vector{F}}; scale_rates = true,
        kwargs...
    ) where {R, F <: Number}
    num_species = size(u0, 1)
    num_nodes = num_sites(spatial_system)
    num_rxs = length(reactstoch)
    @assert size(hopping_constants) == size(u0)
    @assert size(u0, 2) == size(rx_rates, 2) == num_nodes
    @assert length(netstoch) == length(reactstoch) == size(rx_rates, 1) == num_rxs
    spec_CI = CartesianIndices((num_species, num_nodes))
    spec_LI = LinearIndices((num_species, num_nodes))

    sum_outdegrees = sum(outdegree(spatial_system, site) for site in 1:num_nodes)
    num_jumps = num_species * sum_outdegrees + num_nodes * num_rxs
    total_netstoch = R[]
    sizehint!(total_netstoch, num_jumps)
    total_reactstoch = R[]
    sizehint!(total_reactstoch, num_jumps)
    total_rates = F[]
    sizehint!(total_rates, num_jumps)

    #hops
    # assuming hopping_constants isa Matrix{Vector} where hopping_constants[species, src][index] is the hop const of species from src to neighbor at index in neighbors(spatial_system, src)
    for src in 1:num_nodes
        for species in 1:num_species
            hopping_const = hopping_constants[species, src]
            for (i, dst) in enumerate(neighbors(spatial_system, src))
                dst_idx = spec_LI[spec_CI[species, dst]]
                src_idx = spec_LI[spec_CI[species, src]]
                push!(total_netstoch, [dst_idx => 1, src_idx => -1])
                push!(total_reactstoch, [src_idx => 1])
                push!(total_rates, hopping_const[i])
            end
        end
    end

    #reactions
    for site in 1:num_nodes
        for rx in 1:num_rxs
            nstoch = map(p -> Pair(spec_LI[spec_CI[p[1], site]], p[2]), netstoch[rx]) # transform into new indices
            rstoch = map(p -> Pair(spec_LI[spec_CI[p[1], site]], p[2]), reactstoch[rx])
            push!(total_netstoch, nstoch)
            push!(total_reactstoch, rstoch)
        end
    end
    append!(total_rates, vec(rx_rates)) # assuming rx_rates isa Matrix where rx_rates[rx, site] is the rate of rx at site

    # put everything together
    ma_jump = MassActionJump(
        total_rates, total_reactstoch, total_netstoch; nocopy = true,
        scale_rates = scale_rates
    )
    flattened_u0 = vec(u0)
    prob = DiscreteProblem(flattened_u0, tspan, total_rates)
    return prob, ma_jump
end
