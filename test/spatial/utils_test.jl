# Tests for CartesianGrid
using DiffEqJump, LightGraphs
using Test, Random

dims = (4,3,2)
sites = rand(1:prod(dims), 10)
num_samples = 10^5
rel_tol = 0.01
grids = [DiffEqJump.CartesianGridRej(dims), DiffEqJump.CartesianGridIter(dims), LightGraphs.grid(dims)]
for grid in grids
    @test DiffEqJump.num_sites(grid) == prod(dims)
    @test DiffEqJump.num_neighbors(grid, 1) == 3
    @test DiffEqJump.num_neighbors(grid, 4) == 3
    @test DiffEqJump.num_neighbors(grid, 17) == 4
    @test DiffEqJump.num_neighbors(grid, 21) == 3
    @test DiffEqJump.num_neighbors(grid, 6) == 5
    for site in sites
        d = Dict{Int,Int}()
        for i in 1:num_samples
            nb = DiffEqJump.rand_nbr(grid, site)
            nb in keys(d) ? d[nb] += 1 : d[nb] = 1
        end
        for val in values(d)
            @test abs(val/num_samples - 1/DiffEqJump.num_neighbors(grid,site)) < rel_tol
        end
    end
end

# setup
rel_tol = 0.05
num_samples = 10^4
dims = (3,3)
g = grid(dims)
num_nodes = DiffEqJump.num_sites(g)
num_species = 3
reactstoch = [[1 => 1, 2 => 1],[3 => 1]]
netstoch = [[1 => -1, 2 => -1, 3 => 1],[1 => 1, 2 => 1, 3 => -1]]
rates = [0.1, 1.]
num_rxs = length(rates)
ma_jumps = MassActionJump(rates, reactstoch, netstoch)
u = ones(Int, num_species, num_nodes)
rng = MersenneTwister()

# Tests for RxRates
rx_rates = DiffEqJump.RxRates(num_nodes, ma_jumps)
for site in 1:num_nodes
    DiffEqJump.update_rx_rates!(rx_rates, 1:num_rxs, u, site)
    rx_props = [DiffEqJump.evalrxrate(u[:, site], rx, ma_jumps) for rx in 1:num_rxs]
    rx_probs = rx_props/sum(rx_props)
    d = Dict{Int,Int}()
    for i in 1:num_samples
        rx = DiffEqJump.sample_rx_at_site(rx_rates, site, rng)
        rx in keys(d) ? d[rx] += 1 : d[rx] = 1
    end
    for (k,v) in d
        @test abs(v/num_samples - rx_probs[k]) < rel_tol
    end
end

# Tests for HopRatesUnifNbr
hopping_constants = ones(num_species, num_nodes)
hop_rates = DiffEqJump.HopRatesUnifNbr(hopping_constants)
spec_probs = ones(num_species)/num_species

for site in 1:num_nodes
    DiffEqJump.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
    num_nbs = DiffEqJump.num_neighbors(g, site)
    target_probs = ones(num_nbs)/num_nbs
    d1 = Dict{Int,Int}()
    d2 = Dict{Int,Int}()
    for i in 1:num_samples
        spec, target = DiffEqJump.sample_hop_at_site(hop_rates, site, rng, g)
        d1[spec] = get(d1, spec, 0) + 1
        d2[target] = get(d2, target, 0) + 1
    end
    @test maximum(abs.(collect(values(d1))/num_samples - spec_probs)) < rel_tol
    @test maximum(abs.(collect(values(d2))/num_samples - target_probs)) < rel_tol
end

# Tests for HopRatesGeneral
hop_constants = Matrix{Vector{Float64}}(undef, num_species, num_nodes)
for ci in CartesianIndices(hop_constants)
    (species, site) = Tuple(ci)
    hop_constants[ci] = repeat([1.0], DiffEqJump.num_neighbors(g, site))
end
spec_probs = ones(num_species)/num_species
hop_rates = DiffEqJump.HopRatesGeneral(hop_constants)

for site in 1:num_nodes
    DiffEqJump.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
    num_nbs = DiffEqJump.num_neighbors(g, site)
    target_probs = ones(num_nbs)/num_nbs
    d1 = Dict{Int,Int}()
    d2 = Dict{Int,Int}()
    for i in 1:num_samples
        spec, target = DiffEqJump.sample_hop_at_site(hop_rates, site, rng, g)
        d1[spec] = get(d1, spec, 0) + 1
        d2[target] = get(d2, target, 0) + 1
    end
    @test maximum(abs.(collect(values(d1))/num_samples - spec_probs)) < rel_tol
    @test maximum(abs.(collect(values(d2))/num_samples - target_probs)) < rel_tol
end
