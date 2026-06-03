# Tests for topology.jl
using JumpProcesses, Graphs, Test, Random, StableRNGs
const JP = JumpProcesses

# Functions to test:
# num_sites(grid)
# neighbors(grid, site)
# outdegree(grid, site)
# rand_nbr(rng, grid, site)
# nth_nbr(grid, site, n)

io = IOBuffer()
rng = StableRNG(12345)
dims = (4, 3, 2)
sites = 1:prod(dims)
num_samples = 10^5
rel_tol = 0.01
grids = [
    JP.CartesianGridRej(dims),
    Graphs.grid(dims),
]
for grid in grids
    show(io, "text/plain", grid)
    @test String(take!(io)) !== nothing
    @test JP.num_sites(grid) == prod(dims)
    @test JP.outdegree(grid, 1) == 3
    @test JP.outdegree(grid, 4) == 3
    @test JP.outdegree(grid, 17) == 4
    @test JP.outdegree(grid, 21) == 3
    @test JP.outdegree(grid, 6) == 5
    for site in sites
        @test [JP.nth_nbr(grid, site, n) for n in 1:outdegree(grid, site)] ==
            collect(neighbors(grid, site))
        d = Dict{Int, Int}()
        for i in 1:num_samples
            nb = JP.rand_nbr(rng, grid, site)
            nb in keys(d) ? d[nb] += 1 : d[nb] = 1
        end
        for val in values(d)
            @test abs(val / num_samples - 1 / JP.outdegree(grid, site)) < rel_tol
        end
    end
end
