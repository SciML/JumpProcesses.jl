"""
A file with structs and functions for sampling hops and updating hopping rates
"""

# TODO to simplify the design can do two things:
# 1. make an abstract type HopRatesGridOptim (grid-optimized) and dispatch on it. Or maybe dispatch on the type of hop_const_cumulative_sums instead (Array{F} vs Array{Vector{F}})
# 2. make an abstract type HopRatesGraphDsi, which has hop rates uniform wrt neighbors, and treat things not of that type as having hop_const_cumulative_sums field
# 3. Use traits

### spatial hop rates ###
abstract type AbstractHopRates end

function HopRates(hopping_constants::Vector{F}, spatial_system) where {F <: Number}
    HopRatesGraphDs(hopping_constants, num_sites(spatial_system))
end
function HopRates(hopping_constants::Vector{F},
        grid::CartesianGridRej) where {F <: Number}
    HopRatesGraphDs(hopping_constants, num_sites(grid))
end

function HopRates(hopping_constants::Matrix{F}, spatial_system) where {F <: Number}
    HopRatesGraphDsi(hopping_constants)
end
function HopRates(hopping_constants::Matrix{F},
        grid::CartesianGridRej) where {F <: Number}
    HopRatesGraphDsi(hopping_constants)
end

function HopRates(hopping_constants::Matrix{Vector{F}}, spatial_system) where {F <: Number}
    HopRatesGraphDsij(hopping_constants)
end
function HopRates(hopping_constants::Matrix{Vector{F}},
        grid::CartesianGridRej) where {F <: Number}
    HopRatesGridDsij(hopping_constants, grid)
end

function HopRates(p::Pair{SpecHop, SiteHop},
        spatial_system) where {F <: Number, SpecHop <: Vector{F},
        SiteHop <: Vector{Vector{F}}}
    HopRatesGraphDsLij(p...)
end
function HopRates(p::Pair{SpecHop, SiteHop},
        grid::CartesianGridRej) where
        {F <: Number, SpecHop <: Vector{F}, SiteHop <: Vector{Vector{F}}}
    HopRatesGridDsLij(p..., grid)
end

function HopRates(p::Pair{SpecHop, SiteHop},
        spatial_system) where {F <: Number, SpecHop <: Matrix{F},
        SiteHop <: Vector{Vector{F}}}
    HopRatesGraphDsiLij(p...)
end
function HopRates(p::Pair{SpecHop, SiteHop},
        grid::CartesianGridRej) where
        {SpecHop <: Matrix{F}, SiteHop <: Vector{Vector{F}}} where {F <: Number}
    HopRatesGridDsiLij(p..., grid)
end

"""
    update_hop_rates!(hop_rates::AbstractHopRates, species::AbstractArray, u, site, spatial_system)

update rates of all specs in species at site
"""
function update_hop_rates!(hop_rates::AbstractHopRates, species::AbstractArray, u, site,
        spatial_system)
    @inbounds for spec in species
        update_hop_rate!(hop_rates, spec, u, site, spatial_system)
    end
end

hop_rate(hop_rates, species, site) = @inbounds hop_rates.rates[species, site]

"""
    update_hop_rate!(hop_rates::HopRatesGraphDsi, species, u, site, spatial_system)

update rates of single species at site
"""
function update_hop_rate!(hop_rates::AbstractHopRates, species, u, site, spatial_system)
    rates = hop_rates.rates
    @inbounds old_rate = rates[species, site]
    @inbounds rates[species, site] = evalhoprate(hop_rates, u, species, site,
        spatial_system)
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
function sample_species(hop_rates::AbstractHopRates, site, rng)
    @inbounds linear_search((@view hop_rates.rates[:, site]),
        rand(rng) * total_site_hop_rate(hop_rates, site))
end

"""
    sample_hop_at_site(hop_rates::AbstractHopRates, site, rng, spatial_system)

sample a reaction at site, return (species, target_site)
"""
function sample_hop_at_site(hop_rates::AbstractHopRates, site, rng, spatial_system)
    species = sample_species(hop_rates, site, rng)
    return species, sample_target_site(hop_rates, site, species, rng, spatial_system)
end

############## hopping rates of form D_s ################

struct HopRatesGraphDs{F} <: AbstractHopRates
    "hopping_constants[i] is the hop constant of species i"
    hopping_constants::Vector{F}

    "rates[i,j] is total hopping rate of species i at site j"
    rates::Matrix{F}

    "sum_rates[j] is the sum of hopping rates at site j"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGraphDs)
    num_specs, num_sites = size(hop_rates.rates)
    println(io,
        "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_{s} where s is species.")
end

"""
    HopRatesGraphDs(hopping_constants::Vector{F}, num_nodes) where F <: Number

initializes HopRatesGraphDs with zero rates
"""
function HopRatesGraphDs(hopping_constants::Vector{F}, num_nodes) where {F <: Number}
    rates = zeros(F, length(hopping_constants), num_nodes)
    HopRatesGraphDs{F}(hopping_constants, rates, zeros(F, size(rates, 2)))
end

function sample_target_site(hop_rates::HopRatesGraphDs, site, species, rng, spatial_system)
    rand_nbr(rng, spatial_system, site)
end

"""
return hopping rate of species at site
"""
function evalhoprate(hop_rates::HopRatesGraphDs, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.hopping_constants[species] *
              outdegree(spatial_system, site)
end

############## hopping rates of form D_{s,i} ################
struct HopRatesGraphDsi{F} <: AbstractHopRates
    "hopping_constants[i,j] is the hop constant of species i at site j"
    hopping_constants::Matrix{F}

    "rates[i,j] is total hopping rate of species i at site j"
    rates::Matrix{F}

    "sum_rates[j] is the sum of hopping rates at site j"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGraphDsi)
    num_specs, num_sites = size(hop_rates.rates)
    println(io,
        "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_{s,i} where s is species, and i is source.")
end

"""
    HopRatesGraphDsi(hopping_constants::Matrix{F}) where F <: Number

initializes HopRatesGraphDsi with zero rates
"""
function HopRatesGraphDsi(hopping_constants::Matrix{F}) where {F <: Number}
    rates = zeros(F, size(hopping_constants))
    HopRatesGraphDsi{F}(hopping_constants, rates, zeros(F, size(rates, 2)))
end

function sample_target_site(hop_rates::HopRatesGraphDsi, site, species, rng, spatial_system)
    rand_nbr(rng, spatial_system, site)
end

"""
return hopping rate of species at site
"""
function evalhoprate(hop_rates::HopRatesGraphDsi, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.hopping_constants[species, site] *
              outdegree(spatial_system, site)
end

############## hopping rates of form D_{s,i,j} ################
struct HopRatesGraphDsij{F} <: AbstractHopRates
    "hop_const_cumulative_sums[s,i] is the vector of cumulative sums of hopping constants of species s at site i"
    hop_const_cumulative_sums::Matrix{Vector{F}}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGraphDsij)
    num_specs, num_sites = size(hop_rates.rates)
    println(io,
        "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_{s,i,j} where s is species, i is source and j is destination.")
end

"""
    HopRatesGraphDsij(hopping_constants::Matrix{Vector{F}}) where F <: Number

initializes HopRates with zero rates
"""
function HopRatesGraphDsij(hopping_constants::Matrix{Vector{F}};
        do_cumsum = true) where {F <: Number}
    do_cumsum && (hopping_constants = map(cumsum, hopping_constants))
    rates = zeros(F, size(hopping_constants))
    sum_rates = zeros(F, size(rates, 2))
    HopRatesGraphDsij{F}(hopping_constants, rates, sum_rates)
end

function sample_target_site(hop_rates::HopRatesGraphDsij, site, species, rng,
        spatial_system)
    @inbounds cum_hop_consts = hop_rates.hop_const_cumulative_sums[species, site]
    @inbounds n = searchsortedfirst(cum_hop_consts, rand(rng) * cum_hop_consts[end])
    return nth_nbr(spatial_system, site, n)
end

function evalhoprate(hop_rates::HopRatesGraphDsij, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.hop_const_cumulative_sums[species, site][end]
end

#################  hopping rates of form L_{s,i,j} optimized for cartesian grid  ######################
"""
Analogue of HopRatesGraphDsij, optimized for CartesianGrid
"""
struct HopRatesGridDsij{F} <: AbstractHopRates
    "hop_const_cumulative_sums[:,s,i] is the vector of cumulative sums of hopping constants of species s at site i. Out-of-bounds neighbors are treated as having zero propensity."
    hop_const_cumulative_sums::Array{F, 3}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGridDsij)
    num_specs, num_sites = size(hop_rates.rates)
    println(io,
        "HopRates with $num_specs species and $num_sites sites, optimized for CartesianGrid. \nHopping constants of form L_{s,i,j} where s is species, i is source and j is destination.")
end

"""
    HopRatesGridDsij(hopping_constants::Matrix{Vector{F}}) where F <: Number

initializes HopRates with zero rates
"""
function HopRatesGridDsij(hopping_constants::Array{F, 3};
        do_cumsum = true) where {F <: Number}
    do_cumsum && (hopping_constants = mapslices(cumsum, hopping_constants, dims = 1))
    rates = zeros(F, size(hopping_constants)[2:3])
    sum_rates = zeros(F, size(rates, 2))
    HopRatesGridDsij{F}(hopping_constants, rates, sum_rates)
end

function HopRatesGridDsij(hopping_constants::Matrix{Vector{F}}, grid) where {F <: Number}
    new_hopping_constants = Array{F, 3}(undef, 2 * dimension(grid),
        size(hopping_constants)...)
    for ci in CartesianIndices(hopping_constants)
        species, site = Tuple(ci)
        nb_constants = @view new_hopping_constants[:, species, site]
        pad_hop_vec!(nb_constants, grid, site, hopping_constants[ci])
        cumsum!(nb_constants, nb_constants)
    end
    HopRatesGridDsij(new_hopping_constants, do_cumsum = false)
end

function sample_target_site(hop_rates::HopRatesGridDsij, site, species, rng, grid)
    @inbounds cum_hop_consts = @view hop_rates.hop_const_cumulative_sums[:, species, site]
    @inbounds n = searchsortedfirst(cum_hop_consts, rand(rng) * cum_hop_consts[end])
    return nth_potential_nbr(grid, site, n)
end

function evalhoprate(hop_rates::HopRatesGridDsij, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.hop_const_cumulative_sums[end, species, site]
end

############## hopping rates of form D_s * L_{i,j} ################
struct HopRatesGraphDsLij{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Vector{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Vector{Vector{F}}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGraphDsLij)
    num_specs, num_sites = length(hop_rates.species_hop_constants),
    length(hop_rates.hop_const_cumulative_sums)
    println(io,
        "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_s * L_{i,j} where s is species, i is source and j is destination.")
end

"""
    HopRatesGraphDsLij(species_hop_constants::Vector{F}, site_hop_constants::Vector{Vector{F}}) where F <: Number

initializes HopRates with zero rates
"""
function HopRatesGraphDsLij(species_hop_constants::Vector{F},
        site_hop_constants::Vector{Vector{F}};
        do_cumsum = true) where {F <: Number}
    do_cumsum && (site_hop_constants = map(cumsum, site_hop_constants))
    rates = zeros(F, length(species_hop_constants), length(site_hop_constants))
    sum_rates = zeros(size(rates, 2))
    HopRatesGraphDsLij{F}(species_hop_constants, site_hop_constants, rates, sum_rates)
end

function sample_target_site(hop_rates::HopRatesGraphDsLij, site, species, rng,
        spatial_system)
    @inbounds cum_hop_consts = hop_rates.hop_const_cumulative_sums[site]
    @inbounds n = searchsortedfirst(cum_hop_consts, rand(rng) * cum_hop_consts[end])
    return nth_nbr(spatial_system, site, n)
end

function evalhoprate(hop_rates::HopRatesGraphDsLij, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species] *
              hop_rates.hop_const_cumulative_sums[site][end]
end

############## hopping rates of form D_s * L_{i,j} optimized for cartesian grid ################
struct HopRatesGridDsLij{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Vector{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Matrix{F}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGridDsLij)
    num_specs, num_sites = length(hop_rates.species_hop_constants),
    size(hop_rates.hop_const_cumulative_sums, 2)
    println(io,
        "HopRates with $num_specs species and $num_sites sites, optimized for CartesianGrid. \nHopping constants of form D_s * L_{i,j} where s is species, i is source and j is destination.")
end

function HopRatesGridDsLij(species_hop_constants::Vector{F}, site_hop_constants::Matrix{F};
        do_cumsum = true) where {F <: Number}
    do_cumsum && (site_hop_constants = mapslices(cumsum, site_hop_constants, dims = 1))
    rates = zeros(F, length(species_hop_constants), size(site_hop_constants, 2))
    sum_rates = zeros(size(rates, 2))
    HopRatesGridDsLij{F}(species_hop_constants, site_hop_constants, rates, sum_rates)
end

function HopRatesGridDsLij(species_hop_constants::Vector{F},
        site_hop_constants::Vector{Vector{F}}, grid) where {F <: Number}
    new_hopping_constants = Matrix{F}(undef, 2 * dimension(grid),
        length(site_hop_constants))
    for site in 1:length(site_hop_constants)
        nb_constants = @view new_hopping_constants[:, site]
        pad_hop_vec!(nb_constants, grid, site, site_hop_constants[site])
        cumsum!(nb_constants, nb_constants)
    end
    HopRatesGridDsLij(species_hop_constants, new_hopping_constants, do_cumsum = false)
end

function sample_target_site(hop_rates::HopRatesGridDsLij, site, species, rng, grid)
    @inbounds cum_hop_consts = @view hop_rates.hop_const_cumulative_sums[:, site]
    @inbounds n = searchsortedfirst(cum_hop_consts, rand(rng) * cum_hop_consts[end])
    return nth_potential_nbr(grid, site, n)
end

function evalhoprate(hop_rates::HopRatesGridDsLij, u, species, site, grid)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species] *
              hop_rates.hop_const_cumulative_sums[end, site]
end

############## hopping rates of form D_{s,i} * L_{i,j} ################
struct HopRatesGraphDsiLij{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Matrix{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Vector{Vector{F}}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGraphDsiLij)
    num_specs, num_sites = size(hop_rates.species_hop_constants)
    println(io,
        "HopRates with $num_specs species and $num_sites sites. \nHopping constants of form D_{s,i} * L_{i,j} where s is species, i is source and j is destination.")
end

function HopRatesGraphDsiLij(species_hop_constants::Matrix{F},
        site_hop_constants::Vector{Vector{F}};
        do_cumsum = true) where {F <: Number}
    @assert size(species_hop_constants, 2) == length(site_hop_constants)
    do_cumsum && (site_hop_constants = map(cumsum, site_hop_constants))
    rates = zeros(F, length(species_hop_constants), length(site_hop_constants))
    sum_rates = zeros(size(rates, 2))
    HopRatesGraphDsiLij{F}(species_hop_constants, site_hop_constants, rates, sum_rates)
end

function sample_target_site(hop_rates::HopRatesGraphDsiLij, site, species, rng,
        spatial_system)
    @inbounds cum_hop_consts = hop_rates.hop_const_cumulative_sums[site]
    @inbounds n = searchsortedfirst(cum_hop_consts, rand(rng) * cum_hop_consts[end])
    return nth_nbr(spatial_system, site, n)
end

function evalhoprate(hop_rates::HopRatesGraphDsiLij, u, species, site, spatial_system)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species, site] *
              hop_rates.hop_const_cumulative_sums[site][end]
end

############## hopping rates of form D_{s,i} * L_{i,j} optimized for cartesian grid ################
struct HopRatesGridDsiLij{F} <: AbstractHopRates
    "hopping constants of species -- D_s"
    species_hop_constants::Matrix{F}

    "nbs_cumulative[i] is the vector of cumulative sums of hopping constants from site i to its neighbors"
    hop_const_cumulative_sums::Matrix{F}

    "rates[s,i] is the total hopping rate of species s at site i"
    rates::Matrix{F}

    "sum_rates[i] is the total hopping rate out of site i"
    sum_rates::Vector{F}
end

function Base.show(io::IO, ::MIME"text/plain", hop_rates::HopRatesGridDsiLij)
    num_specs, num_sites = length(hop_rates.species_hop_constants),
    size(hop_rates.hop_const_cumulative_sums, 2)
    println(io,
        "HopRates with $num_specs species and $num_sites sites, optimized for CartesianGrid. \nHopping constants of form D_{s,i} * L_{i,j} where s is species, i is source and j is destination.")
end

function HopRatesGridDsiLij(
        species_hop_constants::Matrix{F}, site_hop_constants::Matrix{F};
        do_cumsum = true) where {F <: Number}
    @assert size(species_hop_constants, 2) == size(site_hop_constants, 2)
    do_cumsum && (site_hop_constants = mapslices(cumsum, site_hop_constants, dims = 1))
    rates = zeros(F, size(species_hop_constants))
    sum_rates = zeros(size(rates, 2))
    HopRatesGridDsiLij{F}(species_hop_constants, site_hop_constants, rates, sum_rates)
end

function HopRatesGridDsiLij(species_hop_constants::Matrix{F},
        site_hop_constants::Vector{Vector{F}}, grid) where {F <: Number}
    new_hopping_constants = Matrix{F}(undef, 2 * dimension(grid),
        length(site_hop_constants))
    for site in 1:length(site_hop_constants)
        nb_constants = @view new_hopping_constants[:, site]
        pad_hop_vec!(nb_constants, grid, site, site_hop_constants[site])
        cumsum!(nb_constants, nb_constants)
    end
    HopRatesGridDsiLij(species_hop_constants, new_hopping_constants, do_cumsum = false)
end

function sample_target_site(hop_rates::HopRatesGridDsiLij, site, species, rng, grid)
    @inbounds cum_hop_consts = @view hop_rates.hop_const_cumulative_sums[:, site]
    @inbounds n = searchsortedfirst(cum_hop_consts, rand(rng) * cum_hop_consts[end])
    return nth_potential_nbr(grid, site, n)
end

function evalhoprate(hop_rates::HopRatesGridDsiLij, u, species, site, grid)
    @inbounds u[species, site] * hop_rates.species_hop_constants[species, site] *
              hop_rates.hop_const_cumulative_sums[end, site]
end
