"""
A file with structs and functions for sampling reactions and updating reaction rates in spatial SSAs
"""

### spatial rx rates ###
abstract type AbstractRxRates end

RxRates(coefficients::Nothing, spatial_system, majump::MassActionJump) = RxRatesRs(num_sites(spatial_system), majump)
RxRates(coefficients::Matrix{F}, spatial_system, majump::MassActionJump) where F <: Number = RxRatesRsi(coefficients, majump)

"number of reactions"
num_rxs(rx_rates::AbstractRxRates) = get_num_majumps(rx_rates.ma_jumps)

"""
    reset!(rx_rates::AbstractRxRates)

make all rates zero
"""
function reset!(rx_rates::AbstractRxRates)
    fill!(rx_rates.rates, zero(eltype(rx_rates.rates)))
    fill!(rx_rates.sum_rates, zero(eltype(rx_rates.sum_rates)))
    nothing
end

"""
    total_site_rx_rate(rx_rates::AbstractRxRates, site)

return total reaction rate at site
"""
function total_site_rx_rate(rx_rates::AbstractRxRates, site)
    @inbounds rx_rates.sum_rates[site]
end

"""
sample_rx_at_site(rx_rates::AbstractRxRates, site, rng)

sample a reaction at site, return reaction index
"""
function sample_rx_at_site(rx_rates::AbstractRxRates, site, rng)
    linear_search((@view rx_rates.rates[:,site]), rand(rng) * total_site_rx_rate(rx_rates, site))
end

# helper functions
function set_rx_rate_at_site!(rx_rates::AbstractRxRates, site, rx, rate)
    @inbounds old_rate = rx_rates.rates[rx, site]
    @inbounds rx_rates.rates[rx, site] = rate
    @inbounds rx_rates.sum_rates[site] += rate - old_rate
    old_rate
end

############## reaction rates of form R_i ################
struct RxRatesRs{F,M} <: AbstractRxRates
    "rx_rates[i,j] is rate of reaction i at site j"
    rates::Matrix{F} 

    "rx_rates_sum[j] is sum of reaction rates at site j"
    sum_rates::Vector{F} 
    
    "MassActionJump"
    ma_jumps::M
end

"""
    RxRates(num_sites::Int, ma_jumps::M) where {M}

initializes RxRates with zero rates
"""
function RxRatesRs(num_sites::Int, ma_jumps::M) where {M}
    numrxjumps = get_num_majumps(ma_jumps)
    rates = zeros(Float64, numrxjumps, num_sites)
    RxRatesRs{Float64,M}(rates, vec(sum(rates, dims=1)), ma_jumps)
end

"""
    update_rx_rates!(rx_rates::RxRatesRs, rxs, u, site)

update rates of all reactions in rxs at site
"""
function update_rx_rates!(rx_rates::RxRatesRs, rxs, u, site)
    ma_jumps = rx_rates.ma_jumps
    @inbounds for rx in rxs
        set_rx_rate_at_site!(rx_rates, site, rx, evalrxrate((@view u[:,site]), rx, ma_jumps))
    end
end

function Base.show(io::IO, ::MIME"text/plain", rx_rates::RxRatesRs)
    num_rxs, num_sites = size(rx_rates.rates)
    println(io, "RxRates with $num_rxs reactions and $num_sites sites. \nReaction rates of form D_i where i is reaction.")
end

### spatially varying rx rates ###
struct RxRatesRsi{F,M} <: AbstractRxRates
    "coefficients[i,j] is the coefficient of reaction i at site j by which the base rate is multiplied"
    coefficients::Matrix{F}

    "rx_rates[i,j] is rate of reaction i at site j"
    rates::Matrix{F} 

    "rx_rates_sum[j] is sum of reaction rates at site j"
    sum_rates::Vector{F} 
    
    "MassActionJump"
    ma_jumps::M
end

"""
    RxRatesRsi(num_sites::Int, ma_jumps::M) where {M}

initializes RxRatesRsi with zero rates
"""
function RxRatesRsi(coefficients::Matrix{F}, ma_jumps::M) where {F <: Number, M}
    num_sites = size(coefficients, 2)
    numrxjumps = get_num_majumps(ma_jumps)
    @assert size(coefficients, 1) == numrxjumps
    rates = zeros(F, numrxjumps, num_sites)
    RxRatesRsi{F,M}(coefficients, rates, zeros(F, num_sites), ma_jumps)
end

"""
    update_rx_rates!(rx_rates::RxRatesRsi, rxs, u, site)

update rates of all reactions in rxs at site
"""
function update_rx_rates!(rx_rates::RxRatesRsi, rxs, u, site)
    ma_jumps = rx_rates.ma_jumps
    @inbounds for rx in rxs
        set_rx_rate_at_site!(rx_rates, site, rx, rx_rates.coefficients[rx,site]*evalrxrate((@view u[:,site]), rx, ma_jumps))
    end
end

function Base.show(io::IO, ::MIME"text/plain", rx_rates::RxRatesRsi)
    num_rxs, num_sites = size(rx_rates.rates)
    println(io, "RxRates with $num_rxs reactions and $num_sites sites. \nReaction rates of form D_{i,j} where i is reaction, and j is site.")
end