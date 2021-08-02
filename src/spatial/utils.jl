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
nth_nbr(graph::AbstractGraph, site, n) = neighbors(graph, site)[n]

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
function nth_nbr(grid, site, n)
    CI = grid.CI; offsets = grid.offsets
    @inbounds I = CI[site]
    @inbounds for off in offsets
        nb = I + off
        if nb in CI
            n -= 1
            @inbounds n == 0 && return grid.LI[nb]
        end
    end
end
"""
return neighbors of site in increasing order
"""
function neighbors(grid, site)
    CI = grid.CI
    LI = grid.LI
    I = CI[site]
    Iterators.map(off -> LI[off+I], Iterators.filter(off -> off+I in CI, grid.offsets))
end

# possible rand_nbr functions:
# 1. Rejection-based: pick a neighbor, check it's valid; if not, repeat
# 2. Iterator-based: draw a random number from 1 to num_neighbors and iterate to that neighbor
# 3. Array-based: using a pre-allocated array in grid

# rejection-based
struct CartesianGridRej{N,T}
    dims::NTuple{N, Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
end
function CartesianGridRej(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = [count(x -> x+CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGridRej(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGridRej(dims) = CartesianGridRej(Tuple(dims))
CartesianGridRej(dimension, linear_size::Int) = CartesianGridRej([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGridRej, site::Int)
    CI = grid.CI; offsets = grid.offsets
    @inbounds I = CI[site]
    while true
        @inbounds nb = rand(offsets) + I
        @inbounds nb in CI && return grid.LI[nb]
    end
end

# iterator-based
struct CartesianGridIter{N,T}
    dims::NTuple{N, Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
end
function CartesianGridIter(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = [count(x -> x+CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGridIter(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGridIter(dims) = CartesianGridIter(Tuple(dims))
CartesianGridIter(dimension, linear_size::Int) = CartesianGridIter([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGridIter, site::Int)
    nth_nbr(grid, site, rand(1:num_neighbors(grid,site)))
end

function Base.show(io::IO, grid::Union{CartesianGridRej, CartesianGridIter})
    println(io, "A Cartesian grid with dimensions $(grid.dims)")
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
    RxRates{Float64,M}(rates, vec(sum(rates, dims=1)), ma_jumps)
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
    @inbounds rx_rates.sum_rates[site]
end

"""
update rates of all reactions in rxs at site
"""
function update_rx_rates!(rx_rates, rxs, u, site)
    ma_jumps = rx_rates.ma_jumps
    @inbounds for rx in rxs
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
    @inbounds old_rate = rx_rates.rates[rx, site]
    @inbounds rx_rates.rates[rx, site] = rate
    @inbounds rx_rates.sum_rates[site] += rate - old_rate
    old_rate
end

### spatial hop rates ###
abstract type AbstractHopRates end
"""
update rates of all specs in species at site
"""
function update_hop_rates!(hop_rates::AbstractHopRates, species::AbstractArray, u, site, spatial_system)
    @inbounds for spec in species
        update_hop_rate!(hop_rates, spec, u, site, spatial_system)
    end
end

"""
return total hopping rate out of site
"""
total_site_hop_rate(hop_rates::AbstractHopRates, site) = @inbounds hop_rates.sum_rates[site]

############## hopping rates of form L_{s,i} ################

HopRates(hopping_constants::Matrix{F}) where F <: Number = HopRatesUnifNbr(hopping_constants)
HopRates(hopping_constants::AbstractArray) where F <: Number = HopRatesGeneral(hopping_constants)

struct HopRatesUnifNbr{F} <: AbstractHopRates
    hopping_constants::Matrix{F} # hopping_constants[i,j] is the hop constant of species i at site j
    rates::Matrix{F} # rates[i,j] is total hopping rate of species i at site j
    sum_rates::Vector{F} # sum_rates[j] is the sum of hopping rates at site j
end

"""
initializes HopRatesUnifNbr with zero rates
"""
function HopRatesUnifNbr(hopping_constants::Matrix{F}) where F <: Number
    rates = zeros(eltype(hopping_constants), size(hopping_constants))
    HopRatesUnifNbr{F}(hopping_constants, rates, vec(sum(rates, dims=1)))
end

"""
make all rates zero
"""
function reset!(hop_rates::HopRatesUnifNbr)
    fill!(hop_rates.rates, zero(eltype(hop_rates.rates)))
    fill!(hop_rates.sum_rates, zero(eltype(hop_rates.rates)))
    nothing
end

"""
sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::HopRatesUnifNbr, site, rng, spatial_system) 
    species = linear_search(hop_rates_at_site(hop_rates, site), rand(rng) * total_site_hop_rate(hop_rates, site))
    target_site = rand_nbr(spatial_system, site)
    return species, target_site
end

"""
update rates of single species at site
"""
function update_hop_rate!(hop_rates::HopRatesUnifNbr, species::Int, u, site, spatial_system) 
    set_hop_rate_at_site!(hop_rates, site, species, evalhoprate(hop_rates, u, species, site, num_neighbors(spatial_system, site)))
end

# helper functions
"""
returns hopping rates of a site
"""
function hop_rates_at_site(hop_rates::HopRatesUnifNbr, site) 
    @view hop_rates.rates[:,site]
end

"""
sets the rate of hopping at site. Return the old rate
"""
function set_hop_rate_at_site!(hop_rates::HopRatesUnifNbr, site, species, rate) 
    @inbounds old_rate = hop_rates.rates[species, site]
    @inbounds hop_rates.rates[species, site] = rate
    @inbounds hop_rates.sum_rates[site] += rate - old_rate
    old_rate
end

function evalhoprate(hop_rates::HopRatesUnifNbr, u, species, site, num_nbs::Int) 
    @inbounds u[species,site]*hop_rates.hopping_constants[species,site]*num_nbs
end

############## hopping rates of form L_{s,i,j} ################
struct HopRatesGeneral{F} <: AbstractHopRates
    hop_const_cumulative_sums::Matrix{Vector{F}} # hop_const_cumulative_sums[s,i] is the vector of cumulative sums of hopping constants of species s at site i
    rates::Matrix{F} # rates[s,i] is the total hopping rate of species s at site i
    sum_rates::Vector{F} # sum_rates[i] is the total hopping rate out of site i
end

"""
initializes HopRates with zero rates
"""
function HopRatesGeneral(hopping_constants::Matrix{Vector{F}}) where F <: Number
    hop_const_cumulative_sums = map(cumsum, hopping_constants)
    rates = zeros(F, size(hopping_constants))
    sum_rates = vec(sum(rates, dims=1))
    HopRatesGeneral{F}(hop_const_cumulative_sums, rates, sum_rates)
end

"""
make all rates zero
"""
function reset!(hop_rates::HopRatesGeneral{F}) where F <: Number
    hop_rates.rates .= zero(F)
    hop_rates.sum_rates .= zero(F)
    nothing
end

"""
sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::HopRatesGeneral, site, rng, spatial_system) 
    species = linear_search((@view hop_rates.rates[:,site]), rand(rng) * total_site_hop_rate(hop_rates, site))
    cumulative_hop_constants = hop_rates.hop_const_cumulative_sums[species, site]
    n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return species, nth_nbr(spatial_system, site, n)
end

"""
update rates of single species at site
"""
function update_hop_rate!(hop_rates::HopRatesGeneral, species, u, site, spatial_system)
    rates = hop_rates.rates
    @inbounds old_rate = rates[species, site]
    rates[species, site] = u[species, site] * hop_rates.hop_const_cumulative_sums[species, site][end]
    @inbounds hop_rates.sum_rates[site] += rates[species, site] - old_rate
    old_rate
end


######################## helper routines for all spatial SSAs ########################
total_site_rate(rx_rates::RxRates, hop_rates::AbstractHopRates, site) = total_site_hop_rate(hop_rates, site) + total_site_rx_rate(rx_rates, site)

function update_rates_after_reaction!(p, u, site, reaction_id)
    update_rx_rates!(p.rx_rates, p.dep_gr[reaction_id], u, site)
    update_hop_rates!(p.hop_rates, p.jumptovars_map[reaction_id], u, site, p.spatial_system)
end

function update_rates_after_hop!(p, u, source_site, target_site, species)
    update_rx_rates!(p.rx_rates, p.vartojumps_map[species], u, source_site)
    update_hop_rate!(p.hop_rates, species, u, source_site, p.spatial_system)
    
    update_rx_rates!(p.rx_rates, p.vartojumps_map[species], u, target_site)
    update_hop_rate!(p.hop_rates, species, u, target_site, p.spatial_system)
end

"""
update_state!(p, integrator)

updates state based on p.next_jump
"""
function update_state!(p, integrator)
    jump = p.next_jump
    if is_hop(p, jump)
        execute_hop!(integrator, jump.src, jump.dst, jump.jidx)
    else
        rx_index = reaction_id_from_jump(p,jump)
        @inbounds executerx!((@view integrator.u[:,jump.src]), rx_index, p.rx_rates.ma_jumps)
    end
    # save jump that was just exectued
    p.prev_jump = jump
    nothing
end

"""
    is_hop(p, jump)

true if jump is a hop
"""
function is_hop(p, jump)
    jump.jidx <= p.numspecies
end

"""
    execute_hop!(integrator, jump)

documentation
"""
function execute_hop!(integrator, source_site, target_site, species)
    @inbounds integrator.u[species,source_site] -= 1
    @inbounds integrator.u[species,target_site] += 1
end

"""
    reaction_id_from_jump(p,jump)

return reaction id by subtracting the number of hops
"""
function reaction_id_from_jump(p,jump)
    jump.jidx - p.numspecies
end