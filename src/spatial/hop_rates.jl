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