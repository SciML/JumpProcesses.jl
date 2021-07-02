"""
A file with types, structs and functions for spatial simulations
"""

"""
stores info for a spatial jump
"""
struct SpatialJump{J}
    site::J #where the jump happens
    index::Int #diffusion or reaction
    target_site::J #target of diffusive hop
end

############ abstract spatial system struct ##################
"""
Contains all info about the topology of the system
"""
abstract type AbstractSpatialSystem end

"""
returns neighbors of site
"""
function neighbors end

"""
returns total number of sites
"""
function number_of_sites end

"""
return the number of neighbors of a site
"""
function num_neighbors end

# TODO make a graph struct with connectivity list, compare with CartesianGrid

################### CartesianGrid <: AbstractSpatialSystem ########################
#TODO store the number of neighbors for each site or store all neighbors for each site.
struct CartesianGrid <: AbstractSpatialSystem
    dimension::Int
    linear_size::Int #side length of the grid
end

dimension(grid::CartesianGrid) = grid.dimension

number_of_sites(grid) = grid.linear_size^dimension(grid)

is_site(grid,site_id) = site_id >= 1 && site_id <= number_of_sites(grid)

"""
return a generator that iterates over the neighbors of the given site in grid
"""
function neighbors(grid, site_id)
    (site_id + j*grid.linear_size^(i-1) for i in 1:dimension(grid), j in -1:2:1 if is_site(grid,site_id + j*grid.linear_size^(i-1)))
end

function num_neighbors(grid,site_id)
    counter = 0
    for _ in neighbors(grid,site_id)
        counter += 1
    end
    counter
end

"""
a copy of `nth` function from IterTools
"""
function nth_neighbor(grid,site,n)
    #TODO can make this faster?
    xs = neighbors(grid,site)
    #TODO return negative number instead of BoundsError
    n > 0 || throw(BoundsError(xs, n))

    for (i, val) in enumerate(xs)
        i >= n && return val
    end

    # catch iterators with no length but actual finite size less then n
    throw(BoundsError(xs, n))
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
    reaction_rates::Vector{Vector{R}}
    diffusion_rates::Vector{Vector{R}}
    reaction_rates_sum::Vector{R}
    diffusion_rates_sum::Vector{R}
end

# diffusion_rates[i] = [diff_rate_for_spec_1,...,diff_rate_for_spec_end]

# neighbors = [nbs1,...,nbsend]
# diffusion_rates = [rates1,...,ratesend]
# starting_points = [pointers ]


"""
standard constructor, assumes numeric values for rates
"""
function SpatialRates(reaction_rates::Vector{Vector{R}}, diffusion_rates::Vector{Vector{R}}) where {R}
    SpatialRates{R}(reaction_rates, diffusion_rates, [sum(site) for site in reaction_rates], [sum(site) for site in diffusion_rates])
end
#QUESTION are the type annotations here correct?
"""
initializes SpatialRates with zero rates
"""
function SpatialRates(num_jumps::Integer,num_species::Integer,num_sites::Integer)
    reaction_rates = [zeros(Float64, num_jumps) for i in 1:num_sites]
    diffusion_rates = [zeros(Float64, num_species) for i in 1:num_sites]
    SpatialRates(reaction_rates,diffusion_rates)
end

function SpatialRates(ma_jumps, num_species, spatial_system::AbstractSpatialSystem)
    num_sites = number_of_sites(spatial_system)
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

#sets the rate of reaction at site
function set_site_reaction_rate!(spatial_rates, site_id, reaction_id, rate)
    old_rate = spatial_rates.reaction_rates[site_id][reaction_id]
    spatial_rates.reaction_rates[site_id][reaction_id] = rate
    spatial_rates.reaction_rates_sum[site_id] += rate - old_rate
end

#sets the rate of diffusion at site
function set_site_diffusion_rate!(spatial_rates, site_id, species_id, rate)
    old_rate = spatial_rates.diffusion_rates[site_id][species_id]
    spatial_rates.diffusion_rates[site_id][species_id] = rate
    spatial_rates.diffusion_rates_sum[site_id] += rate - old_rate
end


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
