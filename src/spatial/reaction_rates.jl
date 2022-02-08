"""
A file with structs and functions for sampling reactions and updating reaction rates in spatial SSAs
"""

### spatial rx rates ###
struct RxRates{F,M}
    "rx_rates[i,j] is rate of reaction i at site j"
    rates::Matrix{F} 

    "rx_rates_sum[j] is sum of reaction rates at site j"
    sum_rates::Vector{F} 
    
    "SpatialMassActionJump"
    ma_jumps::M
end

"""
    RxRates(num_sites::Int, ma_jumps::M) where {M}

initializes RxRates with zero rates
"""
function RxRates(num_sites::Int, ma_jumps::M)
    numrxjumps = get_num_majumps(ma_jumps)
    rates = zeros(Float64, numrxjumps, num_sites)
    RxRates{Float64,M}(rates, vec(sum(rates, dims=1)), ma_jumps)
end

num_rxs(rx_rates::RxRates) = get_num_majumps(rx_rates.ma_jumps)

"""
    reset!(rx_rates::RxRates)

make all rates zero
"""
function reset!(rx_rates::RxRates)
    fill!(rx_rates.rates, zero(eltype(rx_rates.rates)))
    fill!(rx_rates.sum_rates, zero(eltype(rx_rates.sum_rates)))
    nothing
end

"""
    total_site_rx_rate(rx_rates::RxRates, site)

return total reaction rate at site
"""
function total_site_rx_rate(rx_rates::RxRates, site)
    @inbounds rx_rates.sum_rates[site]
end

"""
    update_rx_rates!(rx_rates, rxs, u, site)

update rates of all reactions in rxs at site
"""
function update_rx_rates!(rx_rates::RxRates{F,MassActionJump}, rxs, u, site)
    ma_jumps = rx_rates.ma_jumps
    @inbounds for rx in rxs
        set_rx_rate_at_site!(rx_rates, site, rx, evalrxrate((@view u[:,site]), rx, ma_jumps))
    end
end

function update_rx_rates!(rx_rates::RxRates{F,SpatialMassActionJump}, rxs, u, site)
    ma_jumps = rx_rates.ma_jumps
    @inbounds for rx in rxs
        set_rx_rate_at_site!(rx_rates, site, rx, evalrxrate((@view u[:,site]), rx, ma_jumps, site))
    end
end

"""
    sample_rx_at_site(rx_rates::RxRates, site, rng)

sample a reaction at site, return reaction index
"""
function sample_rx_at_site(rx_rates::RxRates, site, rng)
    linear_search((@view rx_rates.rates[:,site]), rand(rng) * total_site_rx_rate(rx_rates, site))
end

# helper functions
function set_rx_rate_at_site!(rx_rates::RxRates, site, rx, rate)
    @inbounds old_rate = rx_rates.rates[rx, site]
    @inbounds rx_rates.rates[rx, site] = rate
    @inbounds rx_rates.sum_rates[site] += rate - old_rate
    old_rate
end

function Base.show(io::IO, ::MIME"text/plain", rx_rates::RxRates)
    num_rxs, num_sites = size(rx_rates.rates)
    println(io, "RxRates with $num_rxs reactions and $num_sites sites")
end