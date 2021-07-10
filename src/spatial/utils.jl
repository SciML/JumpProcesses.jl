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

################### CartesianGrid ########################
"""
Cartesian Grid of dimension D
"""
struct CartesianGrid{D}
    linear_sizes::Vector{Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
end

function CartesianGrid(linear_sizes::Vector)
    I = LinearIndices(Tuple(linear_sizes))
    nums_neighbors = zeros(Int, last(I))
    for site in I
        nums_neighbors[site] = length(neighbors(linear_sizes, site))
    end
    CartesianGrid{length(linear_sizes)}(linear_sizes, nums_neighbors)
end

function CartesianGrid(dimension, linear_size::Integer)
    CartesianGrid([linear_size for i in 1:dimension])
end

dimension(grid::CartesianGrid{D}) where D = D
num_sites(grid::CartesianGrid) = prod(grid.linear_sizes)

# to_coordinates(grid::CartesianGrid{1}, site) = site
# to_coordinates(grid::CartesianGrid{2}, site) = (mod1(site, grid.linear_sizes[1]),fld1(site, grid.linear_sizes[1]))
# function to_coordinates(grid::CartesianGrid{3}, site)
#     temp = mod1(site,grid.linear_sizes[1]*grid.linear_sizes[2])
#     (mod1(temp, grid.linear_sizes[1]),fld1(temp, grid.linear_sizes[1]), fld1(site, grid.linear_sizes[1]*grid.linear_sizes[2]))
# end

# from_coordinates(grid::CartesianGrid{1}, x) = x
# from_coordinates(grid::CartesianGrid{2}, (x,y)) = (y-1) * grid.linear_sizes[1] + x
# from_coordinates(grid::CartesianGrid{3}, (x,y,z)) = (y-1) * grid.linear_sizes[1] + x + (z-1)*grid.linear_sizes[1]*grid.linear_sizes[2]

# issite(grid,site_id::Int) = site_id >= 1 && site_id <= num_sites(grid)
function issite(linear_sizes,ci::CartesianIndex)
    length(ci) == length(linear_sizes) || return false
    for (i,c) in enumerate(Tuple(ci))
        1 <= c && c <= linear_sizes[i] || return false
    end
    return true
end

# potential_neighbors(grid::CartesianGrid{1}, x) = [x-1,x+1]
# potential_neighbors(grid::CartesianGrid{2}, (x,y)) = [(x,y-1),(x-1,y),(x+1,y),(x,y+1)]
# potential_neighbors(grid::CartesianGrid{3}, (x,y,z)) = [(x,y,z-1),(x,y-1,z),(x-1,y,z),(x+1,y,z),(x,y+1,z),(x,y,z+1)]

potential_offsets(grid::CartesianGrid{1}) = [-1,1]
potential_offsets(grid::CartesianGrid{2}) = [CartesianIndex(0,-1),CartesianIndex(-1,0),CartesianIndex(1,0),CartesianIndex(0,1)]
potential_offsets(grid::CartesianGrid{3}) = [CartesianIndex(0,0,-1), CartesianIndex(0,-1,0),CartesianIndex(-1,0,0),CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]
function potential_offsets(dimension::Int)
    if dimension==1
        return [CartesianIndex(-1),CartesianIndex(1)]
    elseif dimension==2
        return [CartesianIndex(0,-1),CartesianIndex(-1,0),CartesianIndex(1,0),CartesianIndex(0,1)]
    elseif dimension==3
        return [CartesianIndex(0,0,-1), CartesianIndex(0,-1,0),CartesianIndex(-1,0,0),CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]
    end
    return []
end
"""
return neighbors of site in CartesianGrid
"""
function neighbors(linear_sizes, site)
    I = LinearIndices(Tuple(linear_sizes))
    J = CartesianIndices(Tuple(linear_sizes))
    ci_site = J[site]
    return [I[offset + ci_site] for offset in potential_offsets(length(linear_sizes)) if issite(linear_sizes,offset + ci_site)]
end
neighbors(grid::CartesianGrid, site) = neighbors(grid.linear_sizes, site)

"""
number of neighbors of site
"""
num_neighbors(grid::CartesianGrid, site) = grid.nums_neighbors[site]



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
function SpatialRates(numrxjumps::Integer,num_species::Integer,num_sites::Integer)
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
