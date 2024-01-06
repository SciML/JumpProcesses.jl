using Graphs
dims = (10, 10, 10)
graph = Graphs.grid(dims)
grid = CartesianGrid(dims)
function outdegree2(grid, site)
    CI = grid.CI
    I = CI[site]
    count(off -> off + I in CI, grid.offsets)
end
function outdegree3(grid, site)
    CI = grid.CI
    I = CI[site]
    res = 0
    for off in grid.offsets
        res += (off + I in CI)
    end
    res
end

using BenchmarkTools
function run(f, g, sites = rand(1:num_sites(grid), 10^6))
    for site in sites
        f(g, site)
    end
end
@time run(outdegree, graph)
@time run(outdegree, grid)
@time run(outdegree2, grid)
@time run(outdegree3, grid)
