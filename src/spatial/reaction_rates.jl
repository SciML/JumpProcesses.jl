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