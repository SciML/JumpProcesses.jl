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

#TODO write different CartesianGrid structs for each sampling method, write rand_nbr(grid, site) for each


#TODO use NTuple{N,T} instead of Tuple for linear_sizes. use N instead of length(linear_sizes)
#TODO make all types concrete
"""
Cartesian Grid of dimension D
"""
struct CartesianGrid{D}
    linear_sizes::Tuple #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices
    LI::LinearIndices
    offsets::Vector
    nbs::Vector{Int}
end

const offsets_1D = [CartesianIndex(-1),CartesianIndex(1)]
const offsets_2D = [CartesianIndex(0,-1),CartesianIndex(-1,0),CartesianIndex(1,0),CartesianIndex(0,1)]
const offsets_3D = [CartesianIndex(0,0,-1), CartesianIndex(0,-1,0),CartesianIndex(-1,0,0),CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]

function CartesianGrid(linear_sizes::Tuple)
    dim = length(linear_sizes)
    CI = CartesianIndices(linear_sizes)
    LI = LinearIndices(linear_sizes)
    offsets = potential_offsets(length(linear_sizes))
    nums_neighbors = zeros(Int, last(LI))
    nbs = zeros(Int, 2*dim)
    grid = CartesianGrid{length(linear_sizes)}(linear_sizes, nums_neighbors, CI, LI, offsets, nbs)
    for site in LI
        nums_neighbors[site] = length(neighbors(linear_sizes, site))
    end
    grid
end

CartesianGrid(linear_sizes) = CartesianGrid(Tuple(linear_sizes))
CartesianGrid(dimension, linear_size::Int) = CartesianGrid([linear_size for i in 1:dimension])

num_sites(grid::CartesianGrid) = prod(grid.linear_sizes)
num_neighbors(grid::CartesianGrid, site) = grid.nums_neighbors[site]

function potential_offsets(dimension::Int)
    if dimension==1
        return offsets_1D
    elseif dimension==2
        return offsets_2D
    else # else dimension == 3
        return offsets_3D
    end
end

# just fill grid.nbs with potential neihbors
function neighbors2(grid::CartesianGrid, site::Int)
    LI, nbs = grid.LI, grid.nbs
    I = grid.CI[site]
    for (j,off) in enumerate(grid.offsets)
        nbs[j] = LI[off+I]
    end
    nbs
end

function rejection_rand(nbs, grid)
    CI = grid.CI
    while true
        nb = rand(nbs)
        nb in CI && return nb
    end
end

######## iterator ###########

#TODO pre-allocate a mutable NbsIter struct in CartesianGrid
struct NbsIter
    grid::CartesianGrid
    site::CartesianIndex
end
NbsIter(grid::CartesianGrid, site::Int) = NbsIter(grid, grid.CI[site])
function Base.iterate(iter::NbsIter, state = 1) #state is the index of the current neighbor in the offsets vector
    grid, site = iter.grid, iter.site
    CI, offsets = grid.CI, grid.offsets
    for i in state:length(offsets)
        off = offsets[i]
        off + site in CI && return (grid.LI[off + site], i+1)
    end
    nothing
end
Base.eltype(::Type{NbsIter}) = Int
Base.length(iter::NbsIter) = num_neighbors(iter.grid, grid.LI[iter.site])

import Base.rand
function rand(iter::NbsIter)
    r = rand(1:length(iter))
    for (i,elt) in enumerate(iter)
        r == i && return elt
    end
end

# function rand(iter::Base.Generator)
#     r = rand(1:length(iter))
#     for (i,elt) in enumerate(iter)
#         r == i && return elt
#     end
# end

# another approach, allocates more
# TODO can pre-allocate function x -> grid.LI[x]
nbs_iter2(grid,site) = Iterators.map(x -> grid.LI[x], Iterators.filter(x -> x+grid.CI[site] in grid.CI, grid.offsets))

################################

# use the pre-allocated array nbs in grid
function neighbors1(grid::CartesianGrid, site::Int)
    CI, LI, nbs = grid.CI, grid.LI, grid.nbs
    I = CI[site]
    j = 1
    for off in grid.offsets
        if off + I in CI
            nbs[j] = LI[off+I]
            j += 1
        end
    end
    @view nbs[1:num_neighbors(grid,site)]
end

# QUESTION how can this be the fastest function???
function neighbors(linear_sizes::Tuple, site)
    J = CartesianIndices(linear_sizes)
    I = LinearIndices(linear_sizes)
    I[filter(x -> (x in J), potential_offsets(length(linear_sizes)) .+ Ref(J[site]))]
end
neighbors(grid::CartesianGrid, site) = neighbors(grid.linear_sizes, site)

function neighbors3(grid::CartesianGrid, site::Int)
    CI = grid.CI
    I = CI[site] 
    Ifirst, Ilast = first(CI), last(CI)
    I1 = oneunit(Ifirst)
    grid.LI[filter(J -> sum(abs.(Tuple(J-I)))==1, max(Ifirst, I-I1):min(Ilast, I+I1))]
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
