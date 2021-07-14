# Tests and benchmarks for various neighbor functions
using BenchmarkTools
using Test
linear_sizes = (4,3,2)
grid = CartesianGrid(linear_sizes)
@test neighbors(grid,1) == [2,5,13]
@test neighbors(grid,4) == [3,8,16]
site = 3

sites = rand(1:num_sites(grid), 100)
for site in sites
    @test collect(NbsIter(grid,site)) == neighbors1(grid,site)
    @test neighbors1(grid, site) == neighbors(linear_sizes, site)
    @test neighbors(linear_sizes, site) == neighbors3(grid, site)
end

function benchmark_init(func, grid, sites)
    for site in sites
        func(grid, site)
    end
end

function benchmark_sample(func, grid, sites)
    for site in sites
        rand(func(grid, site))
    end
end

#TODO benchmark with setup in BenchmarkTools

sites = rand(1:num_sites(grid), 10^4)
funcs_to_sample = [NbsIter, neighbors, neighbors1, neighbors3]
sample_benchmarks = Vector{BenchmarkTools.Trial}(undef, length(funcs_to_sample))
for (i,f) in enumerate(funcs_to_sample)
    sample_benchmarks[i] = @benchmark benchmark_sample($f, $grid, $sites)
end

funcs_to_init = [NbsIter, neighbors, neighbors1, neighbors3]
init_benchmarks = Vector{BenchmarkTools.Trial}(undef, length(funcs_to_init))
for (i,f) in enumerate(funcs_to_init)
    init_benchmarks[i] = @benchmark benchmark_init($f, $grid, $sites)
end


# sites = rand(1:num_sites(grid), 10^6)
# benchmark_init(NbsIter, grid, sites)
# benchmark_init(nbs_iter2, grid, sites)
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