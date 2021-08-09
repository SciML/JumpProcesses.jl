"""
A file with structs and functions for sampling hops and updating hopping rates
"""

# TODO to simplify the design can do two things: 
# 1. make an abstract type HopRatesGridOptim (grid-optimized) and dispatch on it. Or maybe dispatch on the type of hop_const_cumulative_sums instead (Array{F} vs Array{Vector{F}})
# 2. make an abstract type HopRatesUnifNbr, which has hop rates uniform wrt neighbors, and treat things not of that type as having hop_const_cumulative_sums field

### spatial hop rates ###
abstract type AbstractHopRates end

HopRates(hopping_constants::Matrix{F}, spatial_system) where F <: Number = HopRatesUnifNbr(hopping_constants)
HopRates(hopping_constants::Matrix{F}, grid::Union{CartesianGridRej, CartesianGridIter}) where F <: Number = HopRatesUnifNbr(hopping_constants)

HopRates(hopping_constants::Matrix{Vector{F}}, spatial_system) where F <: Number = HopRatesGeneral(hopping_constants)
HopRates(hopping_constants::Matrix{Vector{F}}, grid::Union{CartesianGridRej, CartesianGridIter}) where F <: Number = HopRatesGeneralGrid(hopping_constants, grid)

HopRates(p::Pair{A, B}, spatial_system) where {A <: Vector{F}, B <: Vector{Vector{F}}} where F <: Number = HopRatesMult(p...)
HopRates(p::Pair{A, B}, grid::Union{CartesianGridRej, CartesianGridIter}) where {A <: Vector{F}, B <: Vector{Vector{F}}} where F <: Number = HopRatesMultGrid(p..., grid)

HopRates(p::Pair{A, B}, spatial_system) where {A <: Matrix{F}, B <: Vector{Vector{F}}} where F <: Number = HopRatesMultGeneral(p...)
HopRates(p::Pair{A, B}, grid::Union{CartesianGridRej, CartesianGridIter}) where {A <: Matrix{F}, B <: Vector{Vector{F}}} where F <: Number = HopRatesMultGeneralGrid(p..., grid)

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
    update_hop_rate!(hop_rates::HopRatesUnifNbr, species, u, site, spatial_system)

update rates of single species at site
"""
function update_hop_rate!(hop_rates::AbstractHopRates, species, u, site, spatial_system)
    rates = hop_rates.rates
    @inbounds old_rate = rates[species, site]
    @inbounds rates[species, site] = evalhoprate(hop_rates, u, species, site, spatial_system)
    @inbounds hop_rates.sum_rates[site] += rates[species, site] - old_rate
    old_rate
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

"""
    sample_species(hop_rates::AbstractHopRates, site, rng)

sample species to hop from site
"""
sample_species(hop_rates::AbstractHopRates, site, rng) = @inbounds linear_search((@view hop_rates.rates[:,site]), rand(rng) * total_site_hop_rate(hop_rates, site))

"""
    sample_hop_at_site(hop_rates::AbstractHopRates, site, rng, spatial_system)

sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::AbstractHopRates, site, rng, spatial_system) 
    species = sample_species(hop_rates, site, rng)
    return species, sample_target(hop_rates, site, species, rng, spatial_system)
end

############## hopping rates of form D_{s,i} ################
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
    println(io, "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_{s,i} where s is species, and i is source.")
end

"""
    HopRatesUnifNbr(hopping_constants::Matrix{F}) where F <: Number

initializes HopRatesUnifNbr with zero rates
"""
function HopRatesUnifNbr(hopping_constants::Matrix{F}) where F <: Number
    rates = zeros(eltype(hopping_constants), size(hopping_constants))
    HopRatesUnifNbr{F}(hopping_constants, rates, vec(sum(rates, dims=1)))
end

sample_target(hop_rates::HopRatesUnifNbr, site, species, rng, spatial_system) = rand_nbr(spatial_system, site, rng)

"""
return hopping rate of species at site
"""
function evalhoprate(hop_rates::HopRatesUnifNbr, u, species, site, spatial_system)
    @inbounds u[species,site]*hop_rates.hopping_constants[species,site]*outdegree(spatial_system, site)
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

function sample_target(hop_rates::HopRatesGeneral, site, species, rng, spatial_system)
    @inbounds cumulative_hop_constants = hop_rates.hop_const_cumulative_sums[species, site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return nth_nbr(spatial_system, site, n)
end

function evalhoprate(hop_rates::HopRatesGeneral, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.hop_const_cumulative_sums[species, site][end]
end

#################  hopping rates of form L_{s,i,j} optimized for cartesian grid  ######################
"""
Analogue of HopRatesGeneral, optimized for CartesianGrid
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

function sample_target(hop_rates::HopRatesGeneralGrid, site, species, rng, grid)
    @inbounds cumulative_hop_constants = @view hop_rates.hop_const_cumulative_sums[:,species, site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return nth_potential_nbr(grid, site, n)
end

function evalhoprate(hop_rates::HopRatesGeneralGrid, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.hop_const_cumulative_sums[end, species, site][end]
end

############## hopping rates of form D_s * L_{i,j} ################
struct HopRatesMult{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Vector{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Vector{Vector{F}}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesMult)
    num_specs, num_sites = length(hop_rates.species_hop_constants), length(hop_rates.hop_const_cumulative_sums)
    println(io, "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_s * L_{i,j} where s is species, i is source and j is destination.")
end

"""
    HopRatesMult(species_hop_constants::Vector{F}, site_hop_constants::Vector{Vector{F}}) where F <: Number

initializes HopRates with zero rates
"""
function HopRatesMult(species_hop_constants::Vector{F}, site_hop_constants::Vector{Vector{F}}) where F <: Number
    hop_const_cumulative_sums = map(cumsum, site_hop_constants)
    rates = zeros(F, length(species_hop_constants), length(site_hop_constants))
    sum_rates = vec(sum(rates, dims=1))
    HopRatesMult{F}(species_hop_constants, hop_const_cumulative_sums, rates, sum_rates)
end

function sample_target(hop_rates::HopRatesMult, site, species, rng, spatial_system)
    @inbounds cumulative_hop_constants = hop_rates.hop_const_cumulative_sums[site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return nth_nbr(spatial_system, site, n)
end

function evalhoprate(hop_rates::HopRatesMult, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species] * hop_rates.hop_const_cumulative_sums[site][end]
end

############## hopping rates of form D_s * L_{i,j} optimized for cartesian grid ################
struct HopRatesMultGrid{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Vector{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Matrix{F}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesMultGrid)
    num_specs, num_sites = length(hop_rates.species_hop_constants), size(hop_rates.hop_const_cumulative_sums, 2)
    println(io, "HopRates with $num_specs species and $num_sites sites, optimized for CartesianGrid. \nHopping constants of form D_s * L_{i,j} where s is species, i is source and j is destination.")
end

function HopRatesMultGrid(species_hop_constants::Vector{F}, site_hop_constants::Matrix{F}) where F <: Number
    hop_const_cumulative_sums = mapslices(cumsum, site_hop_constants, dims = 1)
    rates = zeros(F, length(species_hop_constants), size(site_hop_constants, 2))
    sum_rates = vec(sum(rates, dims=1))
    HopRatesMultGrid{F}(species_hop_constants, hop_const_cumulative_sums, rates, sum_rates)
end

function HopRatesMultGrid(species_hop_constants::Vector{F}, site_hop_constants::Vector{Vector{F}}, grid) where F <: Number
    new_hopping_constants = Matrix{F}(undef, 2*dimension(grid), length(site_hop_constants))
    for site in 1:length(site_hop_constants)
        new_hopping_constants[:, site] = pad_hop_vec(grid, site, site_hop_constants[site])
    end
    HopRatesMultGrid(species_hop_constants, new_hopping_constants)
end

function sample_target(hop_rates::HopRatesMultGrid, site, species, rng, grid)
    @inbounds cumulative_hop_constants = @view hop_rates.hop_const_cumulative_sums[:, site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return nth_potential_nbr(grid, site, n)
end

function evalhoprate(hop_rates::HopRatesMultGrid, u, species, site, grid)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species] * hop_rates.hop_const_cumulative_sums[end, site]
end

############## hopping rates of form D_{s,i} * L_{i,j} ################
struct HopRatesMultGeneral{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Matrix{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Vector{Vector{F}}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesMultGeneral)
    num_specs, num_sites = size(hop_rates.species_hop_constants)
    println(io, "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_{s,i} * L_{i,j} where s is species, i is source and j is destination.")
end

function HopRatesMultGeneral(species_hop_constants::Matrix{F}, site_hop_constants::Vector{Vector{F}}) where F <: Number
    @assert size(species_hop_constants, 2) == length(site_hop_constants)
    hop_const_cumulative_sums = map(cumsum, site_hop_constants)
    rates = zeros(F, length(species_hop_constants), length(site_hop_constants))
    sum_rates = vec(sum(rates, dims=1))
    HopRatesMultGeneral{F}(species_hop_constants, hop_const_cumulative_sums, rates, sum_rates)
end

function sample_target(hop_rates::HopRatesMultGeneral, site, species, rng, spatial_system)
    @inbounds cumulative_hop_constants = hop_rates.hop_const_cumulative_sums[site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return nth_nbr(spatial_system, site, n)
end

function evalhoprate(hop_rates::HopRatesMultGeneral, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species, site] * hop_rates.hop_const_cumulative_sums[site][end]
end

############## hopping rates of form D_{s,i} * L_{i,j} optimized for cartesian grid ################
struct HopRatesMultGeneralGrid{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Matrix{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Matrix{F}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesMultGeneralGrid)
    num_specs, num_sites = length(hop_rates.species_hop_constants), size(hop_rates.hop_const_cumulative_sums, 2)
    println(io, "HopRates with $num_specs species and $num_sites sites, optimized for CartesianGrid. \nHopping constants of form D_{s,i} * L_{i,j} where s is species, i is source and j is destination.")
end

function HopRatesMultGeneralGrid(species_hop_constants::Matrix{F}, site_hop_constants::Matrix{F}) where F <: Number
    @assert size(species_hop_constants, 2) == size(site_hop_constants, 2)
    hop_const_cumulative_sums = mapslices(cumsum, site_hop_constants, dims = 1)
    rates = zeros(F, size(species_hop_constants))
    sum_rates = vec(sum(rates, dims=1))
    HopRatesMultGeneralGrid{F}(species_hop_constants, hop_const_cumulative_sums, rates, sum_rates)
end

function HopRatesMultGeneralGrid(species_hop_constants::Matrix{F}, site_hop_constants::Vector{Vector{F}}, grid) where F <: Number
    new_hopping_constants = Matrix{F}(undef, 2*dimension(grid), length(site_hop_constants))
    for site in 1:length(site_hop_constants)
        new_hopping_constants[:, site] = pad_hop_vec(grid, site, site_hop_constants[site])
    end
    HopRatesMultGeneralGrid(species_hop_constants, new_hopping_constants)
end

function sample_target(hop_rates::HopRatesMultGeneralGrid, site, species, rng, grid)
    @inbounds cumulative_hop_constants = @view hop_rates.hop_const_cumulative_sums[:, site]
    @inbounds n = searchsortedfirst(cumulative_hop_constants, rand(rng) * cumulative_hop_constants[end])
    return nth_potential_nbr(grid, site, n)
end

function evalhoprate(hop_rates::HopRatesMultGeneralGrid, u, species, site, grid)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species, site] * hop_rates.hop_const_cumulative_sums[end, site]
end