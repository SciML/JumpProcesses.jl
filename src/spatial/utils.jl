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
    @inbounds I = CI[site]
    while true
        @inbounds nb = rand(offsets) + I
        @inbounds nb in CI && return grid.LI[nb]
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
    @inbounds I = CI[site]
    @inbounds for off in offsets
        nb = I + off
        if nb in CI
            r -= 1
            @inbounds r == 0 && return grid.LI[nb]
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
    nbs = zeros(CartesianIndex{dim}, 2*dim)
    CartesianGrid3(dims, nums_neighbors, CI, LI, offsets, nbs)
end
CartesianGrid3(dims) = CartesianGrid3(Tuple(dims))
CartesianGrid3(dimension, linear_size::Int) = CartesianGrid3([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGrid3, site::Int)
    CI = grid.CI; offsets = grid.offsets; nbs = grid.nbs
    @inbounds I = CI[site]
    j = 0
    @inbounds for off in offsets
        nb = I + off
        @inbounds nb in CI && (j += 1; nbs[j] = nb)
    end
    @inbounds grid.LI[nbs[rand(1:j)]]
end

### spatial rx rates ###
struct RxRates{F,M}
    rates::Matrix{F} # rx_rates[i,j] is rate of reaction i at site j
    sum_rates::Vector{F} # rx_rates_sum[j] is sum of reaction rates at site j
    ma_jumps::M # MassActionJump
end

# functions to implement:
"""
initializes RxRates with zero rates
"""
function RxRates(num_sites::Int, ma_jumps::M) where {M}
    numrxjumps = get_num_majumps(ma_jumps)
    rates = zeros(Float64, numrxjumps, num_sites)
    RxRates{Float64,M}(rates, [sum(@view rates[:,i]) for i in 1:num_sites], ma_jumps)
end

num_rxs(rx_rates::RxRates) = get_num_majumps(rx_rates.ma_jumps)

"""
make all rates zero
"""
function reset!(rx_rates::RxRates)
    fill!(rx_rates.rates, zero(eltype(rx_rates.rates)))
    fill!(rx_rates.sum_rates, zero(eltype(rx_rates.sum_rates)))
    nothing
end

"""
return total reaction rate at site
"""
function total_site_rx_rate(rx_rates::RxRates, site)
    rx_rates.sum_rates[site]
end

"""
update rates of all reactions in rxs at site
"""
function update_reaction_rates!(rx_rates, rxs, u, site)
    ma_jumps = rx_rates.ma_jumps
    for rx in rxs
        set_rx_rate_at_site!(rx_rates, site, rx, evalrxrate((@view u[:,site]), rx, ma_jumps))
    end
end

"""
sample a reaction at site, return reaction index
"""
function sample_rx_at_site(rx_rates::RxRates, site, rng)
    linear_search(rx_rates_at_site(rx_rates, site), rand(rng) * total_site_rx_rate(rx_rates, site))
end

# helper functions
function rx_rates_at_site(rx_rates::RxRates, site)
    @view rx_rates.rates[:,site]
end

function set_rx_rate_at_site!(rx_rates::RxRates, site, rx, rate)
    old_rate = rx_rates.rates[rx, site]
    rx_rates.rates[rx, site] = rate
    rx_rates.sum_rates[site] += rate - old_rate
    old_rate
end

### spatial hop rates ###
struct HopRates{R, F, C} # e.g. HopRates{Matrix{F}, F, Matrix{F}} where F <: Number
    rates::R # rates[i,j] is rate of reaction i at site j
    sum_rates::Vector{F} # sum_rates[j] is the sum of reaction rates at site j
    hopping_constants::C # hopping_constants[i,j] is the hop constant of species i at site j
end

"""
update rates of all specs in species at site
"""
function update_hop_rates!(hop_rates, species::AbstractArray, u, site, spatial_system)
    for spec in species
        update_hop_rate!(hop_rates, spec, u, site, spatial_system)
    end
end

"""
return total hopping rate out of site
"""
total_site_hop_rate(hop_rates::HopRates, site) = hop_rates.sum_rates[site]

############## hopping rates of form L_{s,i} ################
"""
initializes HopRates with zero rates
"""
function HopRates(hopping_constants::Matrix{F}) where F <: Number
    rates = zeros(eltype(hopping_constants), size(hopping_constants))
    HopRates{typeof(rates), F, typeof(hopping_constants)}(rates, sum(rates, dims=1), hopping_constants)
end

"""
make all rates zero
"""
function reset!(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}) where F <: Number
    fill!(hop_rates.rates, zero(eltype(hop_rates.rates)))
    fill!(hop_rates.sum_rates, zero(eltype(hop_rates.rates)))
    nothing
end

"""
sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}, site, rng, spatial_system) where F <: Number
    species = linear_search(hop_rates_at_site(hop_rates, site), rand(rng) * total_site_hop_rate(hop_rates, site))
    target_site = rand_nbr(spatial_system, site)
    return species, target_site
end

"""
update rates of single species at site
"""
function update_hop_rate!(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}, species::Int, u, site, spatial_system) where F <: Number
    set_hop_rate_at_site!(hop_rates, site, species, evalhoprate(hop_rates, u, species, site, num_neighbors(spatial_system, site)))
end

# helper functions
"""
returns hopping rates of a site
"""
function hop_rates_at_site(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}, site) where F <: Number
    @view hop_rates.rates[:,site]
end

"""
sets the rate of hopping at site. Return the old rate
"""
function set_hop_rate_at_site!(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}, site, species, rate) where F <: Number
    old_rate = hop_rates.rates[species, site]
    hop_rates.rates[species, site] = rate
    hop_rates.sum_rates[site] += rate - old_rate
    old_rate
end

function evalhoprate(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}, u, species, site, num_nbs::Int) where F <: Number
    u[species,site]*hop_rates.hopping_constants[species,site]*num_nbs
end

############## hopping rates of form L_{s,i,j} ################
# """
# initializes HopRates with zero rates
# """
# function HopRates(hopping_constants::Matrix{Vector{F}}) where F <: Number
#     rates = copy(hopping_constants)
#     # TODO
#     hop_rates = HopRates{typeof(rates), F, typeof(hopping_constants)}(rates, sum(sum.(x),dims=1), hopping_constants)
#     reset!(hop_rates)
# end

# """
# make all rates zero
# """
# function reset!(hop_rates)
#     # TODO
#     nothing
# end

# """
# sample a reaction at site, return (species, target_site)
# """
# function sample_hop_at_site(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}, site, rng, spatial_system) where F <: Number
#     species = linear_search(hop_rates_at_site(hop_rates, site), rand(rng) * total_site_hop_rate(hop_rates, site))
#     target_site = rand_nbr(spatial_system, site)
#     return species, target_site
# end

# """
# update rates of single species at site
# """
# function update_hop_rate!(hop_rates::HopRates{Matrix{F}, F, Matrix{F}}, species::Int, u, site, spatial_system) where F <: Number
#     set_hop_rate_at_site!(hop_rates, site, species, evalhoprate(hop_rates, u, species, site, num_neighbors(spatial_system, site)))
# end
