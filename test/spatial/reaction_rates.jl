using JumpProcesses, Test, Random, StableRNGs
const JP = JumpProcesses

# Functions to test:
# num_rxs
# reset!
# total_site_rx_rate
# update_rx_rates!
# sample_rx_at_site

# Dummy integrator to test update_rx_rates!
struct DummyIntegrator{U,P,T}
    u::U # state
    p::P # parameters
    t::T # time
end

io = IOBuffer()
# setup of A + B <--> C
rel_tol = 0.05
num_samples = 10^4
num_nodes = 27
num_species = 3
reactstoch = [[1 => 1, 2 => 1], [3 => 1]]
netstoch = [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]]
rates = [0.1, 1.0]
ma_jumps = MassActionJump(rates, reactstoch, netstoch)
spatial_ma_jumps = SpatialMassActionJump(rates, reactstoch, netstoch)
rate_fn = (u, p, t, site) -> 1.0
affect_fn!(integrator) = nothing # a dummy reaction, does nothing
cr_jumps = [ConstantRateJump(rate_fn, affect_fn!)]
num_rxs = 3
u = ones(Int, num_species, num_nodes)
integrator = DummyIntegrator(u, nothing, nothing)
rng = StableRNG(12345)

# Test constructors
@test JP.RxRates(num_nodes, ma_jumps).ma_jumps == ma_jumps
@test JP.RxRates(num_nodes, spatial_ma_jumps).ma_jumps == spatial_ma_jumps
@test JP.RxRates(num_nodes, cr_jumps).cr_jumps == cr_jumps

# Tests for RxRates
rx_rates_list = [JP.RxRates(num_nodes, ma_jumps, cr_jumps), JP.RxRates(num_nodes, spatial_ma_jumps, cr_jumps)]
for rx_rates in rx_rates_list
    @test JP.num_rxs(rx_rates) == num_rxs
    show(io, "text/plain", rx_rates)
    for site in 1:num_nodes
        JP.update_rx_rates!(rx_rates, 1:num_rxs, integrator, site)
        @test JP.total_site_rx_rate(rx_rates, site) == 2.1
        majump_props = [JP.evalrxrate(u[:, site], rx, ma_jumps) for rx in 1:2]
        rx_props = [majump_props..., 1.0]
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