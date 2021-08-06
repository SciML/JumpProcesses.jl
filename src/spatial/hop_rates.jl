"""
A file with structs and functions for sampling hops and updating hopping rates
"""

### spatial hop rates ###
abstract type AbstractHopRates end

HopRates(hopping_constants::Matrix{F}, spatial_system) where F <: Number = HopRatesUnifNbr(hopping_constants)
HopRates(hopping_constants::Matrix{F}, grid::Union{CartesianGridRej, CartesianGridIter}) where F <: Number = HopRatesUnifNbr(hopping_constants)
HopRates(hopping_constants::AbstractArray, spatial_system) = HopRatesGeneral(hopping_constants)
HopRates(hopping_constants::AbstractArray, grid::Union{CartesianGridRej, CartesianGridIter}) = HopRatesGeneralGrid(hopping_constants, grid)

"""
    update_hop_rates!(hop_rates::AbstractHopRates, species::AbstractArray, u, site, spatial_system)

update rates of all specs in species at site
"""
function update_hop_rates!(hop_rates::AbstractHopRates, species::AbstractArray, u, site, spatial_system)
    @inbounds for spec in species
        update_hop_rate!(hop_rates, spec, u, site, spatial_system)
    end
end

"""
    total_site_hop_rate(hop_rates::AbstractHopRates, site)

return total hopping rate out of site
"""
total_site_hop_rate(hop_rates::AbstractHopRates, site) = @inbounds hop_rates.sum_rates[site]

"""
    reset!(hop_rates::AbstractHopRates)

make all rates zero
"""
function reset!(hop_rates::AbstractHopRates)
    hop_rates.rates .= zero(eltype(hop_rates.rates))
    hop_rates.sum_rates .= zero(eltype(hop_rates.sum_rates))
    nothing
end

############## hopping rates of form L_{s,i} ################


struct HopRatesUnifNbr{F} <: AbstractHopRates
    "hopping_constants[i,j] is the hop constant of species i at site j"
    hopping_constants::Matrix{F}

    "rates[i,j] is total hopping rate of species i at site j"
    rates::Matrix{F}

    "sum_rates[j] is the sum of hopping rates at site j"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesUnifNbr)
    num_specs, num_sites = size(hop_rates.rates)
    println(io, "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form L_{s,i} where s is species, and i is source.")
end

"""
    HopRatesUnifNbr(hopping_constants::Matrix{F}) where F <: Number

initializes HopRatesUnifNbr with zero rates
"""
function HopRatesUnifNbr(hopping_constants::Matrix{F}) where F <: Number
    rates = zeros(eltype(hopping_constants), size(hopping_constants))
    HopRatesUnifNbr{F}(hopping_constants, rates, vec(sum(rates, dims=1)))
end

"""
    sample_hop_at_site(hop_rates::HopRatesUnifNbr, site, rng, spatial_system)

sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::HopRatesUnifNbr, site, rng, spatial_system)
    species = linear_search(hop_rates_at_site(hop_rates, site), rand(rng) * total_site_hop_rate(hop_rates, site))
    target_site = rand_nbr(spatial_system, site)
    return species, target_site
end

"""
    update_hop_rate!(hop_rates::HopRatesUnifNbr, species::Int, u, site, spatial_system)

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
"""
return hopping rate of species at site
"""
function evalhoprate(hop_rates::HopRatesUnifNbr, u, species, site, num_nbs::Int)
    @inbounds u[species,site]*hop_rates.hopping_constants[species,site]*num_nbs
end

############## hopping rates of form L_{s,i,j} ################
struct HopRatesGeneral{F} <: AbstractHopRates
    "hop_const_cumulative_sums[s,i] is the vector of cumulative sums of hopping constants of species s at site i"
    hop_const_cumulative_sums::Matrix{Vector{F}}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGeneral)
    num_specs, num_sites = size(hop_rates.rates)
    println(io, "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form L_{s,i,j} where s is species, i is source and j is destination.")
end

"""
    HopRatesGeneral(hopping_constants::Matrix{Vector{F}}) where F <: Number

initializes HopRates with zero rates
"""
function HopRatesGeneral(hopping_constants::Matrix{Vector{F}}) where F <: Number
    hop_const_cumulative_sums = map(cumsum, hopping_constants)
    rates = zeros(F, size(hopping_constants))
    sum_rates = vec(sum(rates, dims=1))
    HopRatesGeneral{F}(hop_const_cumulative_sums, rates, sum_rates)
end

"""
    sample_hop_at_site(hop_rates::HopRatesGeneral, site, rng, spatial_system)

sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::HopRatesGeneral, site, rng, spatial_system)
    @inbounds species = linear_search((@view hop_rates.rates[:,site]), rand(rng) * total_site_hop_rate(hop_rates, site))
    @inbounds cumulative_hop_constants = hop_rates.hop_const_cumulative_sums[species, site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return species, nth_nbr(spatial_system, site, n)
end

"""
    update_hop_rate!(hop_rates::HopRatesGeneral, species, u, site, spatial_system)

update rates of single species at site
"""
function update_hop_rate!(hop_rates::HopRatesGeneral, species, u, site, spatial_system)
    rates = hop_rates.rates
    @inbounds old_rate = rates[species, site]
    @inbounds rates[species, site] = u[species, site] * hop_rates.hop_const_cumulative_sums[species, site][end]
    @inbounds hop_rates.sum_rates[site] += rates[species, site] - old_rate
    old_rate
end

#################  HopRatesGeneralGrid  ######################
"""
Analogue of HopRatesGeneral but for CartesianGrid
"""
struct HopRatesGeneralGrid{F} <: AbstractHopRates
    "hop_const_cumulative_sums[:,s,i] is the vector of cumulative sums of hopping constants of species s at site i"
    hop_const_cumulative_sums::Array{F, 3}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGeneralGrid)
    num_specs, num_sites = size(hop_rates.rates)
    println(io, "HopRates with $num_specs species and $num_sites sites, optimized for CartesianGrid. \nHopping constants of form L_{s,i,j} where s is species, i is source and j is destination.")
end

"""
    HopRatesGeneralGrid(hopping_constants::Matrix{Vector{F}}) where F <: Number

initializes HopRates with zero rates
"""
function HopRatesGeneralGrid(hopping_constants::Array{F, 3}) where F <: Number
    hop_const_cumulative_sums = mapslices(cumsum, hopping_constants, dims = 1)
    rates = zeros(F, size(hopping_constants)[2:3])
    sum_rates = vec(sum(rates, dims=1))
    HopRatesGeneralGrid{F}(hop_const_cumulative_sums, rates, sum_rates)
end

function HopRatesGeneralGrid(hopping_constants::Matrix{Vector{F}}, grid) where F <: Number
    new_hopping_constants = Array{F, 3}(undef, 2*dimension(grid), size(hopping_constants)...)
    for ci in CartesianIndices(hopping_constants)
        species, site = Tuple(ci)
        new_hopping_constants[:, species, site] = pad_hop_vec(grid, site, hopping_constants[ci])
    end
    HopRatesGeneralGrid(new_hopping_constants)
end

"""
    sample_hop_at_site(hop_rates::HopRatesGeneralGrid, site, rng, spatial_system)

sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::HopRatesGeneralGrid, site, rng, grid::Union{CartesianGridRej, CartesianGridIter})
    @inbounds species = linear_search((@view hop_rates.rates[:,site]), rand(rng) * total_site_hop_rate(hop_rates, site))
    @inbounds cumulative_hop_constants = @view hop_rates.hop_const_cumulative_sums[:,species, site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return species, nth_potential_nbr(grid, site, n)
end

"""
    update_hop_rate!(hop_rates::HopRatesGeneralGrid, species, u, site, spatial_system)

update rates of single species at site
"""
function update_hop_rate!(hop_rates::HopRatesGeneralGrid, species, u, site, spatial_system)
    rates = hop_rates.rates
    @inbounds old_rate = rates[species, site]
    @inbounds rates[species, site] = u[species, site] * hop_rates.hop_const_cumulative_sums[end, species, site]
    @inbounds hop_rates.sum_rates[site] += rates[species, site] - old_rate
    old_rate
end

############## hopping rates of form D_s * L_{i,j} ################
# struct HopRatesMult{F} <: AbstractHopRates
#     "cumulative sums of diffusion constants of species"
#     species_cumulative::Vector{F}

#     "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
#     nbs_cumulative::Vector{Vector{F}}

#     "sum_rates[i] is the total hopping rate out of site i"
#     sum_rates::Vector{F}
# end

# function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesMult)
#     num_specs, num_sites = length(species_cumulative), length(nbs_cumulative)
#     println(io, "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_s * L_{i,j} where s is species, i is source and j is destination.")
# end

# """
# HopRatesMult(hopping_constants::Matrix{Vector{F}}) where F <: Number

# initializes HopRates with zero rates
# """
# function HopRatesMult(hopping_constants::Matrix{Vector{F}}) where F <: Number
#     hop_const_cumulative_sums = map(cumsum, hopping_constants)
#     rates = zeros(F, size(hopping_constants))
#     sum_rates = vec(sum(rates, dims=1))
#     HopRatesMult{F}(hop_const_cumulative_sums, rates, sum_rates)
# end

# function reset!(hop_rates::HopRatesMult{F}) where F <: Number
#     hop_rates.sum_rates .= zero(F)
#     nothing
# end

# """
#     sample_hop_at_site(hop_rates::HopRatesMult, site, rng, spatial_system)

# sample a reaction at site, return (species, target_site)
# """
# function sample_hop_at_site(hop_rates::HopRatesMult, site, rng, spatial_system)
#     species = searchsortedfirst(hop_rates.species_cumulative, rand(rng) * hop_rates.species_first_cumulative[end])
#     cumulative_at_site = hop_rates.nbs_cumulative[site]
#     n = searchsortedfirst(cumulative_at_site, rand(rng) * cumulative_at_site[end])
#     return species, nth_nbr(spatial_system, site, n)
# end

# update_hop_rates!(hop_rates::HopRatesMult, species, u, site, spatial_system) = update_hop_rate!(hop_rates, species, u, site, spatial_system)

# """
#     update_hop_rate!(hop_rates::HopRatesMult, species, u, site, spatial_system)

# update rate of site
# """
# function update_hop_rate!(hop_rates::HopRatesMult, species, u, site, spatial_system)
#     old_rate = u[species, site] * hop_rates.nbs_cumulative[site][end]


#     rates[species, site] = u[species, site] * hop_rates.nbs_cumulative[site][end]
#     @inbounds hop_rates.sum_rates[site] += rates[species, site] - old_rate
#     old_rate
# end
