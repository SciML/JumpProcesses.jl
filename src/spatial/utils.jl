"""
A file with types, structs and functions for spatial simulations
"""

"""
stores info for a spatial jump
"""
struct SpatialJump{J}
    src::J    # source location
    jidx::Int # index of jump as a diffusion hop or reaction
    dst::J    # destination location
end

############ abstract spatial system struct ##################
"""
Contains all info about the topology of the system
"""
abstract type AbstractSpatialSystem end

"""
returns total number of sites
"""
function num_sites end

"""
returns neighbors of site
"""
function neighbors end

################### LightGraph ########################
num_sites(graph) = nv(graph)

################### CartesianGrid <: AbstractSpatialSystem ########################
"""
Cartesian Grid of dimension D
"""
struct CartesianGrid{D} <: AbstractSpatialSystem
    linear_sizes::Vector{Int} #side lengths of the grid
end

function CartesianGrid(linear_sizes::Vector)
    CartesianGrid{length(linear_sizes)}(linear_sizes)
end

function CartesianGrid(dimension, linear_size)
    CartesianGrid{dimension}([linear_size for i in 1:dimension])
end

dimension(grid::CartesianGrid{D}) where D = D
num_sites(grid) = prod(grid.linear_sizes)

to_coordinates(grid::CartesianGrid{1}, site) = site
to_coordinates(grid::CartesianGrid{2}, site) = (mod1(site, grid.linear_sizes[1]),fld1(site, grid.linear_sizes[1]))
function to_coordinates(grid::CartesianGrid{3}, site)
    temp = mod1(site,grid.linear_sizes[1]*grid.linear_sizes[2])
    (mod1(temp, grid.linear_sizes[1]),fld1(temp, grid.linear_sizes[1]), fld1(site, grid.linear_sizes[1]*grid.linear_sizes[2]))
end

from_coordinates(grid::CartesianGrid{1}, x) = x
from_coordinates(grid::CartesianGrid{2}, (x,y)) = (y-1) * grid.linear_sizes[1] + x
from_coordinates(grid::CartesianGrid{3}, (x,y,z)) = (y-1) * grid.linear_sizes[1] + x + (z-1)*grid.linear_sizes[1]*grid.linear_sizes[2]

is_site(grid,site_id::Int) = site_id >= 1 && site_id <= num_sites(grid)
function is_site(grid,site_coordinates::Tuple)
    length(p) == dimension(grid) || return false
    for (i,c) in enumerate(site_coordinates)
        1 <= c && c <= grid.linear_sizes[i] || return false
    end
    return true
end

# TODO make these iterators so they don't allocate memory
potential_neighbors(grid::CartesianGrid{1}, x) = [x-1,x+1]
potential_neighbors(grid::CartesianGrid{2}, (x,y)) = [(x,y-1),(x-1,y),(x+1,y),(x,y+1)]
potential_neighbors(grid::CartesianGrid{3}, (x,y,z)) = [(x,y,z-1),(x,y-1,z),(x-1,y,z),(x+1,y,z),(x,y+1,z),(x,y,z+1)]

"""
return neighbors of site in CartesianGrid
"""
function neighbors(grid::CartesianGrid, site::Int)
    [from_coordinates(grid, nb) for nb in potential_neighbors(grid,to_coordinates(grid,site)) if is_site(grid, nb)]
end

#TODO use the Sampler + rand interface to draw a random neighbor as described here https://docs.julialang.org/en/v1/stdlib/Random/#Random.Sampler.

#TODO dealing with escaping:
# make function that returns the ith neighbor for a site -- this might be the sink state in case of absorbing
# or just don't update the target site if it's not a valid site

################### abstract spatial rates struct ###############

abstract type AbstractSpatialRates end

#TODO refactor names of functions?

#return rate of site
function get_site_rate(spatial_rates_struct, site_id)
    get_site_reactions_rate(spatial_rates_struct,site_id)+get_site_diffusions_rate(spatial_rates_struct,site_id)
end

#return rate of reactions at site
function get_site_reactions_rate end

#return rate of diffusions at site
function get_site_diffusions_rate end

#returns an iterator over the pairs (rx_id,rx_rate) of a site
function get_site_reactions_iterator end

#returns an iterator over the pairs (species_id, diffusion_rate) of a site
function get_site_diffusions_iterator end

#sets the rate of reaction at site
function set_site_reaction_rate! end

#sets the rate of diffusion at site
function set_site_diffusion_rate! end

#################### SpatialRates <: AbstractSpatialRates ######################

# NOTE: for now assume diffusion rate only depends on the site and species (and not the neighbor of the site)
# TODO make these matrices instead of vectors of vectors for better performance
# TODO refactor names
# TODO add doc strings to explain what these are
struct SpatialRates{R} <: AbstractSpatialRates
    reaction_rates::Vector{Vector{R}} # [[reaction rates for site 1],...,[reaction rates for last site]]
    diffusion_rates::Vector{Vector{R}} # [[diffusion rates for site 1],...,[diffusion rates for last site]]
    reaction_rates_sum::Vector{R} # [sum of reaction rates for site 1,...,sum of reaction rates for last site]
    diffusion_rates_sum::Vector{R} # [sum of diffusion rates for site 1,...,sum of diffusion rates for last site]
end

"""
standard constructor, assumes numeric values for rates
"""
function SpatialRates(reaction_rates::Vector{Vector{R}}, diffusion_rates::Vector{Vector{R}}) where {R}
    SpatialRates{R}(reaction_rates, diffusion_rates, [sum(site) for site in reaction_rates], [sum(site) for site in diffusion_rates])
end
#QUESTION are the type annotations here correct? Do I need them at all?
"""
initializes SpatialRates with zero rates
"""
function SpatialRates(numrxjumps::Integer,num_species::Integer,num_sites::Integer)
    reaction_rates = [zeros(Float64, numrxjumps) for i in 1:num_sites]
    diffusion_rates = [zeros(Float64, num_species) for i in 1:num_sites]
    SpatialRates(reaction_rates,diffusion_rates)
end

"""
initializes SpatialRates with zero rates
"""
function SpatialRates(ma_jumps, num_species, spatial_system::AbstractSpatialSystem)
    num_sites = num_sites(spatial_system)
    num_jumps = get_num_majumps(ma_jumps)
    SpatialRates(num_jumps,num_species,num_sites)
end

"""
return rate of reactions at site
"""
function get_site_reactions_rate(spatial_rates, site_id)
    spatial_rates.reaction_rates_sum[site_id]
end

"""
return rate of diffusions at site
"""
function get_site_diffusions_rate(spatial_rates, site_id)
    spatial_rates.diffusion_rates_sum[site_id]
end

"""
returns an iterator over reaction rates of a site
"""
function get_site_reactions_iterator(spatial_rates, site_id)
    spatial_rates.reaction_rates[site_id]
end

"""
returns an iterator over diffusion rates of a site
"""
function get_site_diffusions_iterator(spatial_rates, site_id)
    spatial_rates.diffusion_rates[site_id]
end

"""
set the rate of reaction at site. Return the old rate
"""
function set_site_reaction_rate!(spatial_rates, site_id, reaction_id, rate)
    old_rate = spatial_rates.reaction_rates[site_id][reaction_id]
    spatial_rates.reaction_rates[site_id][reaction_id] = rate
    spatial_rates.reaction_rates_sum[site_id] += rate - old_rate
    old_rate
end

"""
sets the rate of diffusion at site. Return the old rate
"""
function set_site_diffusion_rate!(spatial_rates, site_id, species_id, rate)
    old_rate = spatial_rates.diffusion_rates[site_id][species_id]
    spatial_rates.diffusion_rates[site_id][species_id] = rate
    spatial_rates.diffusion_rates_sum[site_id] += rate - old_rate
    old_rate
end


# Tests for CartesianGrid
# using Test
# grid = CartesianGrid([4,3,2])
# for site in 1:length(num_sites(grid))
#     @test from_coordinates(grid,to_coordinates(grid,site)) == site
# end


# Tests for SpatialRates
# using Test
# num_jumps = 2
# num_species = 3
# num_sites = 5
# spatial_rates = DiffEqJump.SpatialRates(num_jumps, num_species, num_sites)

# set_site_reaction_rate!(spatial_rates, 1, 1, 10.0)
# set_site_reaction_rate!(spatial_rates, 1, 1, 20.0)
# set_site_diffusion_rate!(spatial_rates, 1, 1, 30.0)
# @test get_site_reactions_rate(spatial_rates, 1) == 20.0
# @test get_site_diffusions_rate(spatial_rates, 1) == 30.0
