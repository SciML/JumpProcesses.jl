################## Spatial bracketing ##################

# struct to store brackets like (ulow, uhigh), (rx_rates_low, rx_rates_high), (hop_rates_low, hop_rates_high), (site_rates_low, site_rates_high)
struct LowHigh{T}
    low::T
    high::T

    function LowHigh(low::T, high::T; do_copy = true) where {T} 
        if do_copy
            return new{T}(deepcopy(low), deepcopy(high))
        else
            return new{T}(low, high)
        end
    end
    LowHigh(pair::Tuple{T,T}; kwargs...) where {T} = LowHigh(pair[1], pair[2]; kwargs...)
    LowHigh(low_and_high::T; kwargs...) where {T} = LowHigh(low_and_high, low_and_high; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", low_high::LowHigh)
    println(io, "Low: \n $(low_high.low)")
    println(io, "High: \n $(low_high.high)")
end

@inline function update_u_brackets!(u_low_high::LowHigh, bracket_data, u::AbstractMatrix)
    num_species, num_sites = size(u)
    update_u_brackets!(u_low_high, bracket_data, u, 1:num_species, 1:num_sites)
end

@inline function update_u_brackets!(u_low_high::LowHigh, bracket_data, u::AbstractMatrix, species_vec, sites)
    @inbounds for site in sites
        for species in species_vec
            u_low_high[species, site] = LowHigh(get_spec_brackets(bracket_data, species, u[species, site]))
        end
    end
    nothing
end

function is_inside_brackets(u_low_high::LowHigh{M}, u::M, species, site) where {M}
    return u_low_high.low[species, site] < u[species, site] < u_low_high.high[species, site]
end

### convenience functions for LowHigh ###
function setindex!(low_high::LowHigh{A}, val::LowHigh, i...) where {A <: AbstractArray}
    low_high.low[i...] = val.low
    low_high.high[i...] = val.high
    val
end
getindex(low_high::LowHigh{A}, i) where {A <: AbstractArray} = LowHigh(low_high.low[i], low_high.high[i])

get_majumps(rx_rates::LowHigh{R}) where {R <: RxRates} = get_majumps(rx_rates.low)

function total_site_rate(rx_rates::LowHigh, hop_rates::LowHigh, site)
    return LowHigh(
        total_site_rate(rx_rates.low, hop_rates.low, site),
        total_site_rate(rx_rates.high, hop_rates.high, site))
end

function update_rx_rates!(rx_rates::LowHigh, rxs, u_low_high, integrator, site)
    update_rx_rates!(rx_rates.low, rxs, u_low_high.low, integrator, site)
    update_rx_rates!(rx_rates.high, rxs, u_low_high.high, integrator, site)
end

function update_hop_rates!(hop_rates::LowHigh, species, u_low_high, site, spatial_system)
    update_hop_rates!(hop_rates.low, species, u_low_high.low, site, spatial_system)
    update_hop_rates!(hop_rates.high, species, u_low_high.high, site, spatial_system)
end

function reset!(low_high::LowHigh) 
    reset!(low_high.low)
    reset!(low_high.high)
end