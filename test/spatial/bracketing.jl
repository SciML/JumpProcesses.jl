using JumpProcesses, Test
const JP = JumpProcesses

fluctuation_rate = 0.1
threshold = 25
Δu = 4
bracket_data = BracketData(fluctuation_rate, threshold, Δu)

n = 3 # number of sites

# set up spatial system
spatial_system = CartesianGrid((n,)) # n sites

# set up reaction rates
majump_rates = [0.1] # death at rate 0.1
reactstoch = [[1 => 1]]
netstoch = [[1 => -1]]
majump = MassActionJump(majump_rates, reactstoch,
    netstoch)
rx_rates = JP.LowHigh(JP.RxRates(n, majump))

# set up hop rates
hop_constants = [1.0]
hop_rates = JP.LowHigh(JP.HopRates(hop_constants, spatial_system))

# set up species brackets
u = 100 * ones(Int, 1, n) # 2 species, n sites
u_low_high = JP.LowHigh(u, u)
JP.update_u_brackets!(u_low_high, bracket_data, u)

# update reaction rates, hop rates and site rates
rxs = [1] # vector of all reactions
species_vec = [1] # vector of all species
integrator = Nothing # only needed for constant rate jumps
for site in 1:num_sites(spatial_system)
    JP.update_rx_rates!(rx_rates, rxs, u_low_high, integrator, site)
    JP.update_hop_rates!(hop_rates, species_vec, u_low_high, site, spatial_system)
end

# test species brackets
@test u_low_high.low[1, 1]≈u[1, 1] * (1 - fluctuation_rate) atol=1
@test u_low_high.high[1, 1]≈u[1, 1] * (1 + fluctuation_rate) atol=1

# test site rate brackets
site = 1
rx = 1
species = 1
@test JP.total_site_rx_rate(rx_rates.low, site) ==
      majump_rates[rx] * u_low_high.low[species, site]
@test JP.total_site_rx_rate(rx_rates.high, site) ==
      majump_rates[rx] * u_low_high.high[species, site]
@test JP.total_site_hop_rate(hop_rates.low, site) ==
      hop_constants[site] * u_low_high.low[species, site]
@test JP.total_site_hop_rate(hop_rates.high, site) ==
      hop_constants[site] * u_low_high.high[species, site]
