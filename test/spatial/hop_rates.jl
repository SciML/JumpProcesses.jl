using JumpProcesses, Graphs, Test, Random, StableRNGs
const JP = JumpProcesses

# Functions to test
# HopRates(hop_constants, spatial_system)
# update_hop_rates!(hop_rates::AbstractHopRates, species::AbstractArray, u, site, spatial_system)
# total_site_hop_rate(hop_rates::AbstractHopRates, site)
# reset!(hop_rates::AbstractHopRates)
# sample_hop_at_site(hop_rates::AbstractHopRates, site, rng, spatial_system)

function test_reset(hop_rates, num_nodes)
    JP.reset!(hop_rates)
    for site in 1:num_nodes
        @test JP.total_site_hop_rate(hop_rates, site) == 0.0
    end
    return
end

function normalized(distribution::Dict)
    d = copy(distribution)
    s = sum(values(d))
    for key in keys(d)
        d[key] /= s
    end
    return d
end

normalized(distribution) = distribution / sum(distribution)

function statistical_test(
        hop_rates, spec_propensities, target_propensities::Dict,
        num_species, u, site, g, rng, rel_tol
    )
    spec_probs = normalized(spec_propensities)
    target_probs = normalized(target_propensities)
    JP.update_hop_rates!(hop_rates, 1:num_species, u, site, g)
    @test JP.total_site_hop_rate(hop_rates, site) == sum(spec_propensities)
    spec_dict = Dict{Int, Int}()
    site_dict = Dict{Int, Int}()
    for _ in 1:num_samples
        spec, target = JP.sample_hop_at_site(hop_rates, site, rng, g)
        spec_dict[spec] = get(spec_dict, spec, 0) + 1
        site_dict[target] = get(site_dict, target, 0) + 1
    end
    for species in 1:num_species
        @test abs(spec_dict[species]) / num_samples - spec_probs[species] < rel_tol
    end
    for target in JP.neighbors(g, site)
        @test abs(site_dict[target]) / num_samples - target_probs[target] < rel_tol
    end
    return
end

io = IOBuffer()
rng = StableRNG(12345)
rel_tol = 0.05
num_samples = 10^4
dims = (3, 2)
g = CartesianGridRej(dims)
num_nodes = JP.num_sites(g)
num_species = 2
u = ones(Int, num_species, num_nodes)

# Tests for HopRatesGraphDs
hop_constants = [1.0, 3.0] # species
hop_rates = JP.HopRates(hop_constants, g)
@test hop_rates isa JP.HopRatesGraphDs
show(io, "text/plain", hop_rates)
for site in 1:num_nodes
    spec_propensities = hop_constants * JP.outdegree(g, site)
    target_propensities = Dict{Int, Float64}()
    for (i, target) in enumerate(JP.neighbors(g, site))
        target_propensities[target] = 1.0
    end
    statistical_test(
        hop_rates, spec_propensities, target_propensities, num_species, u,
        site, g, rng, rel_tol
    )
end
test_reset(hop_rates, num_nodes)

# Tests for HopRatesGraphDsi
hop_constants = [2.0 1.0 1.0 1.0 1.0 1.0; 6.0 3.0 3.0 3.0 3.0 3.0] # [species, site]
hop_rates = JP.HopRates(hop_constants, g)
@test hop_rates isa JP.HopRatesGraphDsi
show(io, "text/plain", hop_rates)
for site in 1:num_nodes
    spec_propensities = hop_constants[:, site] * JP.outdegree(g, site)
    target_propensities = Dict{Int, Float64}()
    for (i, target) in enumerate(JP.neighbors(g, site))
        target_propensities[target] = 1.0
    end
    statistical_test(
        hop_rates, spec_propensities, target_propensities, num_species, u,
        site, g, rng, rel_tol
    )
end
test_reset(hop_rates, num_nodes)

# Tests for HopRatesGraphDsij
hop_constants = Matrix{Vector{Float64}}(undef, num_species, num_nodes) # [species, site][target_site]
hop_constants[1, :] = [
    [2.0, 4.0],
    [1.0, 2.0, 4.0],
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0, 4.0],
    [1.0, 2.0],
]
hop_constants[2, :] = [
    [3.0, 12.0],
    [3.0, 6.0, 12.0],
    [3.0, 6.0],
    [3.0, 6.0],
    [3.0, 6.0, 12.0],
    [3.0, 6.0],
]
hop_rates_structs = [
    JP.HopRatesGraphDsij(hop_constants),
    JP.HopRates(hop_constants, g),
]
@test hop_rates_structs[2] isa JP.HopRatesGridDsij
for hop_rates in hop_rates_structs
    show(io, "text/plain", hop_rates)
    for site in 1:num_nodes
        spec_propensities = [sum(hop_constants[species, site]) for species in 1:num_species]
        target_propensities = Dict{Int, Float64}()
        for (i, target) in enumerate(JP.neighbors(g, site))
            target_propensities[target] = sum(
                [
                    hop_constants[species, site][i]
                        for species in 1:num_species
                ]
            )
        end
        statistical_test(hop_rates, spec_propensities, target_propensities, num_species, u, site, g, rng, rel_tol)
    end
end
test_reset(hop_rates, num_nodes)

# Tests for HopRatesGraphDsLij
species_hop_constants = [1.0, 3.0] #species
site_hop_constants = [
    [2.0, 4.0],
    [1.0, 2.0, 4.0],
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0, 4.0],
    [1.0, 2.0],
] #[site][target_site]
hop_rates_structs = [
    JP.HopRatesGraphDsLij(species_hop_constants, site_hop_constants),
    JP.HopRates((species_hop_constants => site_hop_constants), g),
]
@test hop_rates_structs[2] isa JP.HopRatesGridDsLij
for hop_rates in hop_rates_structs
    show(io, "text/plain", hop_rates)
    for site in 1:num_nodes
        spec_propensities = [
            species_hop_constants[species] * sum(site_hop_constants[site])
                for species in 1:num_species
        ]
        target_propensities = Dict{Int, Float64}()
        for (i, target) in enumerate(JP.neighbors(g, site))
            target_propensities[target] = sum(
                [
                    species_hop_constants[species] *
                        site_hop_constants[site][i]
                        for species in 1:num_species
                ]
            )
        end
        statistical_test(hop_rates, spec_propensities, target_propensities, num_species, u, site, g, rng, rel_tol)
    end
end
test_reset(hop_rates, num_nodes)

# Tests for HopRatesGraphDsiLij
species_hop_constants = [2.0 1.0 1.0 1.0 1.0 1.0; 6.0 3.0 3.0 3.0 3.0 3.0] #[species, site]
site_hop_constants = [
    [2.0, 4.0],
    [1.0, 2.0, 4.0],
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0, 4.0],
    [1.0, 2.0],
] #[site][target_site]
hop_rates_structs = [
    JP.HopRatesGraphDsiLij(species_hop_constants, site_hop_constants),
    JP.HopRates((species_hop_constants => site_hop_constants), g),
]
@test hop_rates_structs[2] isa JP.HopRatesGridDsiLij
for hop_rates in hop_rates_structs
    show(io, "text/plain", hop_rates)
    for site in 1:num_nodes
        spec_propensities = [species_hop_constants[species, site] * sum(site_hop_constants[site]) for species in 1:num_species]
        target_propensities = Dict{Int, Float64}()
        for (i, target) in enumerate(JP.neighbors(g, site))
            target_propensities[target] = sum(
                [
                    species_hop_constants[species, site] *
                        site_hop_constants[site][i]
                        for species in 1:num_species
                ]
            )
        end
        statistical_test(hop_rates, spec_propensities, target_propensities, num_species, u, site, g, rng, rel_tol)
    end
end
test_reset(hop_rates, num_nodes)

@test String(take!(io)) !== nothing
