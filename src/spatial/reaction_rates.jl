"""
A file with structs and functions for sampling reactions and updating reaction rates in spatial SSAs
"""

### spatial rx rates ###
struct RxRates{F, M}
    "rx_rates[i,j] is rate of reaction i at site j"
    rates::Matrix{F}

    "rx_rates_sum[j] is sum of reaction rates at site j"
    sum_rates::Vector{F}

    "AbstractMassActionJump"
    ma_jumps::M
end

"""
    RxRates(num_sites::Int, ma_jumps::M) where {M}

initializes RxRates with zero rates
"""
function RxRates(num_sites::Int, ma_jumps::M) where {M}
    numrxjumps = get_num_majumps(ma_jumps)
    rates = zeros(Float64, numrxjumps, num_sites)
    RxRates{Float64, M}(rates, vec(sum(rates, dims = 1)), ma_jumps)
end

num_rxs(rx_rates::RxRates) = get_num_majumps(rx_rates.ma_jumps)
get_majumps(rx_rates::RxRates) = rx_rates.ma_jumps

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
function update_rx_rates!(rx_rates::RxRates{F, M}, rxs, u::AbstractMatrix, integrator,
    site) where {F, M}
    ma_jumps = rx_rates.ma_jumps
    @inbounds for rx in rxs
        rate = eval_massaction_rate(u, rx, ma_jumps, site)
        set_rx_rate_at_site!(rx_rates, site, rx, rate)
    end
end

function update_rx_rates!(rx_rates::RxRates{F, M}, rxs, integrator,
    site) where {F, M <: AbstractMassActionJump}
    u = integrator.u
    update_rx_rates!(rx_rates, rxs, u, integrator, site)
end

"""
    sample_rx_at_site(rx_rates::RxRates, site, rng)

sample a reaction at site, return reaction index
"""
function sample_rx_at_site(rx_rates::RxRates, site, rng)
    linear_search((@view rx_rates.rates[:, site]),
                  rand(rng) * total_site_rx_rate(rx_rates, site))
end

"""
    recompute_site_rx_rate(rx_rates::RxRates, u, site)

compute the total reaction rate at site at the current state u
"""
function recompute_site_rx_rate(rx_rates::RxRates, u, site)
    rate = zero(eltype(rx_rates.rates))
    ma_jumps = rx_rates.ma_jumps
    for rx in 1:num_rxs(rx_rates)
        rate += eval_massaction_rate(u, rx, ma_jumps, site)
    end
    return rate
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

eval_massaction_rate(u, rx, ma_jumps::M, site) where {M <: SpatialMassActionJump} = evalrxrate(u, rx, ma_jumps, site)
eval_massaction_rate(u, rx, ma_jumps::M, site) where {M <: MassActionJump} = evalrxrate((@view u[:, site]), rx, ma_jumps)
