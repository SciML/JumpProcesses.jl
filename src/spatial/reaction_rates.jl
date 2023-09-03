"""
A file with structs and functions for sampling reactions and updating reaction rates in spatial SSAs.
Massaction jumps go first in the indexing, then constant rate jumps.
"""

### spatial rx rates ###
struct RxRates{F, M, C}
    "rx_rates[i,j] is rate of reaction i at site j"
    rates::Matrix{F}

    "rx_rates_sum[j] is sum of reaction rates at site j"
    sum_rates::Vector{F}

    "AbstractMassActionJump"
    ma_jumps::M

    "indexable collection of SpatialConstantRateJump"
    cr_jumps::C
end

"""
    RxRates(num_sites::Int, ma_jumps::M, cr_jumps::C) where {M, C}

initializes RxRates with zero rates
"""
function RxRates(num_sites::Int, ma_jumps::M, cr_jumps::C) where {M, C}
    numrxjumps = get_num_majumps(ma_jumps) + length(cr_jumps)
    rates = zeros(Float64, numrxjumps, num_sites)
    RxRates{Float64, M, C}(rates, vec(sum(rates, dims = 1)), ma_jumps, cr_jumps)
end
RxRates(num_sites::Int, ma_jumps::M) where {M<:AbstractMassActionJump} = RxRates(num_sites, ma_jumps, ConstantRateJump[])
RxRates(num_sites::Int, cr_jumps::C) where {C} = RxRates(num_sites, SpatialMassActionJump(nothing), cr_jumps)

num_rxs(rx_rates::RxRates) = get_num_majumps(rx_rates.ma_jumps) + length(rx_rates.cr_jumps)

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
    update_rx_rates!(rx_rates, rxs, integrator, site)

update rates of all reactions in rxs at site
"""
function update_rx_rates!(rx_rates::RxRates, rxs, integrator,
                          site)
    u = integrator.u
    ma_jumps = rx_rates.ma_jumps
    @inbounds for rx in rxs
        if is_massaction(rx_rates, rx)
            rate = eval_massaction_rate(u, rx, ma_jumps, site)
            set_rx_rate_at_site!(rx_rates, site, rx, rate)
        else
            cr_jump = rx_rates.cr_jumps[rx - get_num_majumps(ma_jumps)]
            set_rx_rate_at_site!(rx_rates, site, rx, cr_jump.rate(u, integrator.p, integrator.t, site))
        end
    end
end

"""
    sample_rx_at_site(rx_rates::RxRates, site, rng)

sample a reaction at site, return reaction index
"""
function sample_rx_at_site(rx_rates::RxRates, site, rng)
    linear_search((@view rx_rates.rates[:, site]),
                  rand(rng) * total_site_rx_rate(rx_rates, site))
end

function execute_rx_at_site!(integrator, rx_rates::RxRates, rx, site)
    if is_massaction(rx_rates, rx)
        @inbounds executerx!((@view integrator.u[:, site]), rx,
                             rx_rates.ma_jumps)
    else
        cr_jump = rx_rates.cr_jumps[rx - get_num_majumps(rx_rates.ma_jumps)]
        cr_jump.affect!(integrator)
    end
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

"Return true if jump is a massaction jump."
function is_massaction(rx_rates::RxRates, rx)
    rx <= get_num_majumps(rx_rates.ma_jumps)
end

eval_massaction_rate(u, rx, ma_jumps::M, site) where {M <: SpatialMassActionJump} = evalrxrate(u, rx, ma_jumps, site)
eval_massaction_rate(u, rx, ma_jumps::M, site) where {M <: MassActionJump} = evalrxrate((@view u[:, site]), rx, ma_jumps)