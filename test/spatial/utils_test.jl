# Tests for CartesianGrid
using DiffEqJump, LightGraphs
using Test

dims = (4,3,2)
sites = rand(1:prod(dims), 10)
num_samples = 10^5
rel_tol = 0.01
grids = [DiffEqJump.CartesianGrid1(dims), DiffEqJump.CartesianGrid2(dims), DiffEqJump.CartesianGrid3(dims), LightGraphs.grid(dims)]
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
            if abs(val/num_samples - 1/DiffEqJump.num_neighbors(grid,site)) > rel_tol
                @show typeof(grid), site, d
            end
        end
    end
end

# Tests for SpatialRates
num_jumps = 2
num_species = 3
num_nodes = 5
hopping_constants = ones(num_species, num_nodes)
reactstoch = [[1 => 1, 2 => 1],[3 => 1]]
netstoch = [[1 => -1, 2 => -1, 3 => 1],[1 => 1, 2 => 1, 3 => -1]]
rates = [0.1, 1.]
ma_jumps = MassActionJump(rates, reactstoch, netstoch)

rx_rates = DiffEqJump.RxRates(num_nodes, ma_jumps)
hop_rates = DiffEqJump.HopRates(hopping_constants)

DiffEqJump.set_rx_rate_at_site!(rx_rates, 1, 1, 10.0)
DiffEqJump.set_rx_rate_at_site!(rx_rates, 1, 1, 20.0)
DiffEqJump.set_hop_rate_at_site!(hop_rates, 1, 1, 20.0)
DiffEqJump.set_hop_rate_at_site!(hop_rates, 1, 1, 30.0)
@test DiffEqJump.total_site_rx_rate(rx_rates, 1) == 20.0
@test DiffEqJump.total_site_hop_rate(hop_rates, 1) == 30.0