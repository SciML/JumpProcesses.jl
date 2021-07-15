# benchmarks for various grids
using BenchmarkTools, LightGraphs

dims = (4,3,2)
sites = 1:prod(dims)
grids = [CartesianGrid1(dims), CartesianGrid2(dims), CartesianGrid3(dims), LightGraphs.grid(dims)]
benchmarks = Vector{BenchmarkTools.Trial}(undef, length(grids))

for (i,grid) in enumerate(grids)
    benchmarks[i] = @benchmark rand_nbr($grid, site) setup = (site = rand(sites))
end