# Tests for CartesianGrid
using JumpProcesses, Graphs
using Test, Random

io = IOBuffer()
rng = MersenneTwister()
dims = (4, 3, 2)
sites = rand(1:prod(dims), 10)
num_samples = 10^5
rel_tol = 0.01
grids = [
    JumpProcesses.CartesianGridRej(dims),
    JumpProcesses.CartesianGridIter(dims),
    Graphs.grid(dims),
]
for grid in grids
    show(io, "text/plain", grid)
    @test String(take!(io)) !== nothing
    @test JumpProcesses.num_sites(grid) == prod(dims)
    @test JumpProcesses.outdegree(grid, 1) == 3
    @test JumpProcesses.outdegree(grid, 4) == 3
    @test JumpProcesses.outdegree(grid, 17) == 4
    @test JumpProcesses.outdegree(grid, 21) == 3
    @test JumpProcesses.outdegree(grid, 6) == 5
    for site in sites
        d = Dict{Int, Int}()
        for i in 1:num_samples
            nb = JumpProcesses.rand_nbr(rng, grid, site)
            nb in keys(d) ? d[nb] += 1 : d[nb] = 1
        end
        for val in values(d)
            @test abs(val / num_samples - 1 / JumpProcesses.outdegree(grid, site)) < rel_tol
        end
    end
end

# setup
rel_tol = 0.05
num_samples = 10^4
dims = (3, 3, 3)
g = CartesianGridRej(dims)
num_nodes = JumpProcesses.num_sites(g)
num_species = 3
reactstoch = [[1 => 1, 2 => 1], [3 => 1]]
netstoch = [[1 => -1, 2 => -1, 3 => 1], [1 => 1, 2 => 1, 3 => -1]]
rates = [0.1, 1.0]
num_rxs = length(rates)
ma_jumps = MassActionJump(rates, reactstoch, netstoch)
u = ones(Int, num_species, num_nodes)
rng = MersenneTwister()

# Tests for RxRates
rx_rates = JumpProcesses.RxRates(num_nodes, ma_jumps)
show(io, "text/plain", rx_rates)
for site in 1:num_nodes
    JumpProcesses.update_rx_rates!(rx_rates, 1:num_rxs, u, site)
    rx_props = [JumpProcesses.evalrxrate(u[:, site], rx, ma_jumps) for rx in 1:num_rxs]
    rx_probs = rx_props / sum(rx_props)
    d = Dict{Int, Int}()
    for i in 1:num_samples
        rx = JumpProcesses.sample_rx_at_site(rx_rates, site, rng)
        rx in keys(d) ? d[rx] += 1 : d[rx] = 1
    end
    for (k, v) in d
        @test abs(v / num_samples - rx_probs[k]) < rel_tol
    end
end

# Tests for HopRatesGraphDs
hopping_constants = ones(num_species)
hop_rates = JumpProcesses.HopRatesGraphDs(hopping_constants, num_nodes)
show(io, "text/plain", hop_rates)
spec_probs = ones(num_species) / num_species

for site in 1:num_nodes
    JumpProcesses.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
    num_nbs = JumpProcesses.outdegree(g, site)
    target_probs = ones(num_nbs) / num_nbs
    d1 = Dict{Int, Int}()
    d2 = Dict{Int, Int}()
    for i in 1:num_samples
        spec, target = JumpProcesses.sample_hop_at_site(hop_rates, site, rng, g)
        d1[spec] = get(d1, spec, 0) + 1
        d2[target] = get(d2, target, 0) + 1
    end
    @test maximum(abs.(collect(values(d1)) / num_samples - spec_probs)) < rel_tol
    @test maximum(abs.(collect(values(d2)) / num_samples - target_probs)) < rel_tol
end

# Tests for HopRatesGraphDsi
hopping_constants = ones(num_species, num_nodes)
hop_rates = JumpProcesses.HopRatesGraphDsi(hopping_constants)
show(io, "text/plain", hop_rates)
spec_probs = ones(num_species) / num_species

for site in 1:num_nodes
    JumpProcesses.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
    num_nbs = JumpProcesses.outdegree(g, site)
    target_probs = ones(num_nbs) / num_nbs
    d1 = Dict{Int, Int}()
    d2 = Dict{Int, Int}()
    for i in 1:num_samples
        spec, target = JumpProcesses.sample_hop_at_site(hop_rates, site, rng, g)
        d1[spec] = get(d1, spec, 0) + 1
        d2[target] = get(d2, target, 0) + 1
    end
    @test maximum(abs.(collect(values(d1)) / num_samples - spec_probs)) < rel_tol
    @test maximum(abs.(collect(values(d2)) / num_samples - target_probs)) < rel_tol
end

# Tests for HopRatesGraphDsij
hop_constants = Matrix{Vector{Float64}}(undef, num_species, num_nodes)
for ci in CartesianIndices(hop_constants)
    (species, site) = Tuple(ci)
    hop_constants[ci] = repeat([1.0], JumpProcesses.outdegree(g, site))
end
spec_probs = ones(num_species) / num_species
hop_rates_structs = [
    JumpProcesses.HopRatesGraphDsij(hop_constants),
    JumpProcesses.HopRatesGridDsij(hop_constants, g),
]

for hop_rates in hop_rates_structs
    show(io, "text/plain", hop_rates)
    for site in 1:num_nodes
        JumpProcesses.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
        num_nbs = JumpProcesses.outdegree(g, site)
        target_probs = ones(num_nbs) / num_nbs
        d1 = Dict{Int, Int}()
        d2 = Dict{Int, Int}()
        for i in 1:num_samples
            spec, target = JumpProcesses.sample_hop_at_site(hop_rates, site, rng, g)
            d1[spec] = get(d1, spec, 0) + 1
            d2[target] = get(d2, target, 0) + 1
        end
        @test maximum(abs.(collect(values(d1)) / num_samples - spec_probs)) < rel_tol
        @test maximum(abs.(collect(values(d2)) / num_samples - target_probs)) < rel_tol
    end
end

# Tests for HopRatesGraphDsLij
species_hop_constants = ones(num_species)
site_hop_constants = Vector{Vector{Float64}}(undef, num_nodes)
for site in 1:num_nodes
    site_hop_constants[site] = repeat([1.0], JumpProcesses.outdegree(g, site))
end
spec_probs = ones(num_species) / num_species
hop_rates_structs = [
    JumpProcesses.HopRatesGraphDsLij(species_hop_constants, site_hop_constants),
    JumpProcesses.HopRatesGridDsLij(species_hop_constants, site_hop_constants, g),
]

for hop_rates in hop_rates_structs
    show(io, "text/plain", hop_rates)
    for site in 1:num_nodes
        JumpProcesses.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
        num_nbs = JumpProcesses.outdegree(g, site)
        target_probs = ones(num_nbs) / num_nbs
        d1 = Dict{Int, Int}()
        d2 = Dict{Int, Int}()
        for i in 1:num_samples
            spec, target = JumpProcesses.sample_hop_at_site(hop_rates, site, rng, g)
            d1[spec] = get(d1, spec, 0) + 1
            d2[target] = get(d2, target, 0) + 1
        end
        @test maximum(abs.(collect(values(d1)) / num_samples - spec_probs)) < rel_tol
        @test maximum(abs.(collect(values(d2)) / num_samples - target_probs)) < rel_tol
    end
end

# Tests for HopRatesGraphDsiLij
species_hop_constants = ones(num_species, num_nodes)
site_hop_constants = Vector{Vector{Float64}}(undef, num_nodes)
for site in 1:num_nodes
    site_hop_constants[site] = repeat([1.0], JumpProcesses.outdegree(g, site))
end
spec_probs = ones(num_species) / num_species
hop_rates_structs = [
    JumpProcesses.HopRatesGraphDsiLij(species_hop_constants, site_hop_constants),
    JumpProcesses.HopRatesGridDsiLij(species_hop_constants, site_hop_constants, g),
]

for hop_rates in hop_rates_structs
    show(io, "text/plain", hop_rates)
    for site in 1:num_nodes
        JumpProcesses.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
        num_nbs = JumpProcesses.outdegree(g, site)
        target_probs = ones(num_nbs) / num_nbs
        d1 = Dict{Int, Int}()
        d2 = Dict{Int, Int}()
        for i in 1:num_samples
            spec, target = JumpProcesses.sample_hop_at_site(hop_rates, site, rng, g)
            d1[spec] = get(d1, spec, 0) + 1
            d2[target] = get(d2, target, 0) + 1
        end
        @test maximum(abs.(collect(values(d1)) / num_samples - spec_probs)) < rel_tol
        @test maximum(abs.(collect(values(d2)) / num_samples - target_probs)) < rel_tol
    end
end

@test String(take!(io)) !== nothing
