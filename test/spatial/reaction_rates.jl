using JumpProcesses, Test, Random, StableRNGs
const JP = JumpProcesses

# Functions to test:
# num_rxs
# reset!
# total_site_rx_rate
# update_rx_rates!
# update_rx_rates!
# sample_rx_at_site

io = IOBuffer()
# setup of A + B <--> C
rel_tol = 0.05
num_samples = 10^4
num_nodes = 27
num_species = 3
reactstoch = [[1 => 1, 2 => 1], [3 => 1]]
netstoch = [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]]
rates = [0.1, 1.0]
num_rxs = length(rates)
ma_jumps = MassActionJump(rates, reactstoch, netstoch)
spatial_ma_jumps = SpatialMassActionJump(rates, reactstoch, netstoch)
u = ones(Int, num_species, num_nodes)
rng = StableRNG(12345)

# Tests for RxRates
rx_rates_list = [JP.RxRates(num_nodes, ma_jumps), JP.RxRates(num_nodes, spatial_ma_jumps)]
for rx_rates in rx_rates_list
    @test JP.num_rxs(rx_rates) == length(rates)
    show(io, "text/plain", rx_rates)
    for site in 1:num_nodes
        JP.update_rx_rates!(rx_rates, 1:num_rxs, u, site)
        @test JP.total_site_rx_rate(rx_rates, site) == 1.1
        rx_props = [JP.evalrxrate(u[:, site], rx, ma_jumps) for rx in 1:num_rxs]
        rx_probs = rx_props / sum(rx_props)
        d = Dict{Int, Int}()
        for i in 1:num_samples
            rx = JP.sample_rx_at_site(rx_rates, site, rng)
            rx in keys(d) ? d[rx] += 1 : d[rx] = 1
        end
        for (k, v) in d
            @test abs(v / num_samples - rx_probs[k]) < rel_tol
        end
    end
    JP.reset!(rx_rates)
    for site in 1:num_nodes
        @test JP.total_site_rx_rate(rx_rates, site) == 0.0
    end
end
