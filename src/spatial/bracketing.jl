################## Spatial bracketing ##################

# struct to store brackets like (ulow, uhigh), (rx_rates_low, rx_rates_high), (hop_rates_low, hop_rates_high), (site_rates_low, site_rates_high)
struct LowHigh{T}
    low::T
    high::T

    LowHigh(low::T, high::T) where {T} = new{T}(deepcopy(low), deepcopy(high))
    LowHigh(pair::Tuple{T, T}) where {T} = new{T}(pair[1], pair[2])
    LowHigh(low_and_high::T) where {T} = new{T}(low_and_high, deepcopy(low_and_high))
end

function Base.show(io::IO, ::MIME"text/plain", low_high::LowHigh)
    println(io, "Low: \n $(low_high.low)")
    println(io, "High: \n $(low_high.high)")
end

@inline function update_u_brackets!(u_low_high::LowHigh, bracket_data, u::AbstractMatrix)
    @inbounds for (i, uval) in enumerate(u)
        u_low_high[i] = LowHigh(get_spec_brackets(bracket_data, i, uval))
    end
    nothing
end

### convenience functions for LowHigh ###
function setindex!(low_high::LowHigh, val::LowHigh, i)
    low_high.low[i] = val.low
    low_high.high[i] = val.high
    val
end

function getindex(low_high::LowHigh, i)
    return LowHigh(low_high.low[i], low_high.high[i])
end

function total_site_rate(rx_rates::LowHigh, hop_rates::LowHigh, site)
    return LowHigh(total_site_rate(rx_rates.low, hop_rates.low, site),
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
