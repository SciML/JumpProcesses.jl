"""
A file with types, structs and functions for spatial simulations
"""

"""
stores info for a spatial jump
"""
struct SpatialJump{J}
    src::J    # source location
    jidx::Int # index of jump as a hop or reaction
    dst::J    # destination location
end

############ spatial system interface ##################

# num_sites(spatial_system) = total number of sites
# neighbors(spatial_system, site) = an iterator over the neighbors of site
# num_neighbors(spatial_system, site) = number of neighbors of site

################### LightGraph ########################
num_sites(graph::AbstractGraph) = LightGraphs.nv(graph)
# neighbors(graph::AbstractGraph, site) = LightGraphs.neighbors(graph, site)
num_neighbors(graph::AbstractGraph, site) = LightGraphs.outdegree(graph, site)
rand_nbr(graph::AbstractGraph, site) = rand(neighbors(graph, site))

################### CartesianGrid ########################
const offsets_1D = [CartesianIndex(-1),CartesianIndex(1)]
const offsets_2D = [CartesianIndex(0,-1),CartesianIndex(-1,0),CartesianIndex(1,0),CartesianIndex(0,1)]
const offsets_3D = [CartesianIndex(0,0,-1), CartesianIndex(0,-1,0),CartesianIndex(-1,0,0),CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]
"""
dimension is assumed to be 1, 2, or 3
"""
function potential_offsets(dimension::Int)
    if dimension==1
        return offsets_1D
    elseif dimension==2
        return offsets_2D
    else # else dimension == 3
        return offsets_3D
    end
end

num_sites(grid) = prod(grid.dims)
num_neighbors(grid, site) = grid.nums_neighbors[site]


# possible rand_nbr functions:
# 1. Rejection-based: pick a neighbor, check it's valid; if not, repeat
# 2. Iterator-based: draw a random number from 1 to num_neighbors and iterate to that neighbor
# 3. Array-based: using a pre-allocated array in grid

# rejection-based
struct CartesianGrid1{N,T}
    dims::NTuple{N, Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
end
function CartesianGrid1(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = [count(x -> x+CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGrid1(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGrid1(dims) = CartesianGrid1(Tuple(dims))
CartesianGrid1(dimension, linear_size::Int) = CartesianGrid1([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGrid1, site::Int)
    CI = grid.CI; offsets = grid.offsets
    I = CI[site]
    while true
        nb = rand(offsets) + I
        nb in CI && return grid.LI[nb]
    end
end

# custom iterator-based
struct CartesianGrid2{N,T}
    dims::NTuple{N, Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
end
function CartesianGrid2(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = [count(x -> x+CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGrid2(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGrid2(dims) = CartesianGrid2(Tuple(dims))
CartesianGrid2(dimension, linear_size::Int) = CartesianGrid2([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGrid2, site::Int)
    r = rand(1:num_neighbors(grid,site))
    CI = grid.CI; offsets = grid.offsets
    I = CI[site]
    for off in offsets
        nb = I + off
        if nb in CI
            r -= 1
            r == 0 && return grid.LI[nb]
        end
    end
end

# array-based
struct CartesianGrid3{N,T}
    dims::NTuple{N, Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
    nbs::Vector{CartesianIndex{N}}
end
function CartesianGrid3(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = [count(x -> x+CI[site] in CI, offsets) for site in 1:prod(dims)]
    nbs = zeros(CartesianIndex{3}, 2*dim)
    CartesianGrid3(dims, nums_neighbors, CI, LI, offsets, nbs)
end
CartesianGrid3(dims) = CartesianGrid3(Tuple(dims))
CartesianGrid3(dimension, linear_size::Int) = CartesianGrid3([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGrid3, site::Int)
    CI = grid.CI; offsets = grid.offsets; nbs = grid.nbs
    I = CI[site]
    j = 0
    for off in offsets
        nb = I + off
        nb in CI && (j += 1; nbs[j] = nb)
    end
    grid.LI[nbs[rand(1:j)]]
end

################### abstract spatial rates struct ###############

abstract type AbstractSpatialRates end

"""
return total rate of site
"""
function total_site_rate(spatial_rates_struct, site_id)
    total_site_rx_rate(spatial_rates_struct,site_id)+total_site_hop_rate(spatial_rates_struct,site_id)
end

""" 
return total reaction rate of the site
"""
function total_site_rx_rate end

""" 
return total hopping rate of the site
"""
function total_site_hop_rate end

"""
return the reaction rates at the site as an interator
"""
function rx_rates_at_site end

"""
return the hopping rates at the site as an interator
"""
function hop_rates_at_site end

"""
set the rate of reaction at site
"""
function set_rx_rate_at_site! end

"""
set the rate of hopping at site
"""
function set_hop_rate_at_site! end

#################### SpatialRates <: AbstractSpatialRates ######################

# NOTE: for now assume hopping rate only depends on the site and species (and not the neighbor of the site)
struct SpatialRates{R} <: AbstractSpatialRates
    rx_rates::Matrix{R} # rx_rates[i,j] is rate of reaction i at site j
    hop_rates::Matrix{R} # hop_rates[i,j] is rate of species i at site j
    rx_rates_sum::Vector{R} # [sum of reaction rates for site 1,...,sum of reaction rates for last site]
    hop_rates_sum::Vector{R} # [sum of hopping rates for site 1,...,sum of hopping rates for last site]
end

"""
standard constructor, assumes numeric values for rates
"""
function SpatialRates(rx_rates::Matrix{R}, hop_rates::Matrix{R}) where {R}
    num_sites = size(rx_rates, 2)
    SpatialRates{R}(rx_rates, hop_rates, [sum(@view rx_rates[:,i]) for i in 1:num_sites], [sum(@view hop_rates[:,i]) for i in 1:num_sites])
end

"""
initializes SpatialRates with zero rates
"""
function SpatialRates(numrxjumps::Int,num_species::Int,num_sites::Int)
    reaction_rates = zeros(Float64, numrxjumps, num_sites)
    hopping_rates = zeros(Float64, num_species, num_sites)
    SpatialRates(reaction_rates,hopping_rates)
end

"""
initializes SpatialRates with zero rates
"""
function SpatialRates(ma_jumps, num_species, spatial_system)
    num_sites = num_sites(spatial_system)
    num_jumps = get_num_majumps(ma_jumps)
    SpatialRates(num_jumps,num_species,num_sites)
end

"""
make all rates zero
"""
function reset!(spatial_rates)
    fill!(spatial_rates.rx_rates, zero(eltype(spatial_rates.rx_rates)))
    fill!(spatial_rates.hop_rates, zero(eltype(spatial_rates.hop_rates)))
    fill!(spatial_rates.rx_rates_sum, zero(eltype(spatial_rates.rx_rates_sum)))
    fill!(spatial_rates.hop_rates_sum, zero(eltype(spatial_rates.hop_rates_sum)))
    nothing
end
"""
return total reaction rate at site
"""
function total_site_rx_rate(spatial_rates, site_id)
    spatial_rates.rx_rates_sum[site_id]
end

"""
return total hopping rate out of site
"""
function total_site_hop_rate(spatial_rates, site_id)
    spatial_rates.hop_rates_sum[site_id]
end

"""
returns reaction rates of a site
"""
function rx_rates_at_site(spatial_rates, site_id)
    @view spatial_rates.rx_rates[:,site_id]
end

"""
returns hopping rates of a site
"""
function hop_rates_at_site(spatial_rates, site_id)
    @view spatial_rates.hop_rates[:,site_id]
end

"""
set the rate of reaction at site. Return the old rate
"""
function set_rx_rate_at_site!(spatial_rates, site_id, reaction_id, rate)
    old_rate = spatial_rates.rx_rates[reaction_id, site_id]
    spatial_rates.rx_rates[reaction_id, site_id] = rate
    spatial_rates.rx_rates_sum[site_id] += rate - old_rate
    old_rate
end

"""
sets the rate of hopping at site. Return the old rate
"""
function set_hop_rate_at_site!(spatial_rates, site_id, species_id, rate)
    old_rate = spatial_rates.hop_rates[species_id, site_id]
    spatial_rates.hop_rates[species_id, site_id] = rate
    spatial_rates.hop_rates_sum[site_id] += rate - old_rate
    old_rate
end
