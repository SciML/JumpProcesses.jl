# Tests for CartesianGrid
using DiffEqJump
using Test
grid = CartesianGrid([4,3,2])
for site in 1:length(DiffEqJump.num_sites(grid))
    @test DiffEqJump.from_coordinates(grid,DiffEqJump.to_coordinates(grid,site)) == site
end
@test DiffEqJump.neighbors(grid,1) == [2,5,13]
@test DiffEqJump.neighbors(grid,4) == [3,8,16]
@test DiffEqJump.neighbors(grid,17) == [5,13,18,21]
@test DiffEqJump.neighbors(grid,21) == [9,17,22]
@test DiffEqJump.num_neighbors(grid, 1) == 3

# Tests for SpatialRates
using Test
num_jumps = 2
num_species = 3
# num_sites = 5
spatial_rates = DiffEqJump.SpatialRates(num_jumps, num_species, 5)

DiffEqJump.set_rx_rate_at_site!(spatial_rates, 1, 1, 10.0)
DiffEqJump.set_rx_rate_at_site!(spatial_rates, 1, 1, 20.0)
DiffEqJump.set_hop_rate_at_site!(spatial_rates, 1, 1, 30.0)
@test DiffEqJump.total_site_rx_rate(spatial_rates, 1) == 20.0
@test DiffEqJump.total_site_hop_rate(spatial_rates, 1) == 30.0
