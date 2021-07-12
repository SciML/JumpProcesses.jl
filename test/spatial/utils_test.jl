# Tests for CartesianGrid
using DiffEqJump
using Test
grid = CartesianGrid([4,3,2])
# for site in 1:length(DiffEqJump.num_sites(grid))
#     @test DiffEqJump.from_coordinates(grid,DiffEqJump.to_coordinates(grid,site)) == site
# end
@test DiffEqJump.neighbors(grid,1) == [2,5,13]
@test DiffEqJump.neighbors(grid,4) == [3,8,16]
@test DiffEqJump.neighbors(grid,17) == [5,13,18,21]
@test DiffEqJump.neighbors(grid,21) == [9,17,22]
@test DiffEqJump.num_neighbors(grid, 1) == 3

dims = grid.linear_sizes
for site in 1:DiffEqJump.num_sites(grid)
    @test Set(DiffEqJump.neighbors(dims,site)) == Set(DiffEqJump.nbs(dims,site))
end

# Tests for SpatialRates
num_jumps = 2
num_species = 3
# num_sites = 5
spatial_rates = DiffEqJump.SpatialRates(num_jumps, num_species, 5)

DiffEqJump.set_rx_rate_at_site!(spatial_rates, 1, 1, 10.0)
DiffEqJump.set_rx_rate_at_site!(spatial_rates, 1, 1, 20.0)
DiffEqJump.set_hop_rate_at_site!(spatial_rates, 1, 1, 30.0)
@test DiffEqJump.total_site_rx_rate(spatial_rates, 1) == 20.0
@test DiffEqJump.total_site_hop_rate(spatial_rates, 1) == 30.0











# Tests and benchmarks for various neighbor functions
# using BenchmarkTools
# using Test
# linear_sizes = (4,3,2)
# grid = CartesianGrid(linear_sizes)
# @test neighbors(grid,1) == [2,5,13]
# @test neighbors(grid,4) == [3,8,16]

# sites = rand(1:num_sites(grid), 100)
# for site in sites
#     @test collect(NbsIter(grid,site)) == neighbors1(grid,site)
#     @test neighbors1(grid, site) == neighbors(linear_sizes, site)
#     @test neighbors(linear_sizes, site) == neighbors3(grid, site)
# end

# function benchmark_init(func, grid, sites)
#     res = 0.0
#     for site in sites
#         res += @elapsed func(grid, site)
#     end
#     res
# end

# function benchmark_sample(func, grid, sites)
#     res = 0.0
#     for site in sites
#         res += @elapsed rand(func(grid, site))
#     end
#     res
# end

# sites = rand(1:num_sites(grid), 10^6)
# benchmark_init(NbsIter, grid, sites)
# benchmark_init(neighbors1, grid, sites)
# benchmark_init(neighbors, grid, sites)
# benchmark_init(neighbors3, grid, sites)

# sites = rand(1:num_sites(grid), 10^5)
# benchmark_sample(NbsIter, grid, sites)
# benchmark_sample(neighbors1, grid, sites)
# benchmark_sample(neighbors, grid, sites)
# benchmark_sample(neighbors3, grid, sites)



# site = 3
# @btime neighbors1($grid, $site)
# @btime neighbors($grid.linear_sizes, $I)
# @btime neighbors2($grid, $site)