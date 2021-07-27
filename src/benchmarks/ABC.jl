# benchmarks for various grids
using DiffEqJump, BenchmarkTools, LightGraphs
using ProgressLogging

non_spatial_algs = [NRM(), RSSACR(), DirectCR()]
num_probs = length(non_spatial_algs) + 4
solve_ABC_benchmarks = [Vector{BenchmarkTools.Trial}(undef, num_probs), Vector{BenchmarkTools.Trial}(undef, num_probs)]
for (k,dim) in enumerate([1,3])
    linear_size = 32
    dims = Tuple([linear_size for i in 1:dim])
    starting_site = trunc(Int,(prod(dims) + 1)/2)
    u0 = [linear_size^dim,linear_size^dim,0]
    end_time = 0.1
    diffusivity = 1.0

    domain_size = 1.0 #μ-meter
    mesh_size = domain_size/linear_size
    # ABC model A + B <--> C
    reactstoch = [[1 => 1, 2 => 1],[3 => 1]]
    netstoch = [[1 => -1, 2 => -1, 3 => 1],[1 => 1, 2 => 1, 3 => -1]]
    rates = [0.1/mesh_size, 1.]
    majumps = MassActionJump(rates, reactstoch, netstoch)

    # spatial system setup
    hopping_rate = diffusivity / mesh_size^2
    num_nodes = prod(dims)

    # Starting state setup
    starting_state = zeros(Int, length(u0), num_nodes)
    starting_state[:,starting_site] .= u0
    prob = DiscreteProblem(starting_state,(0.0,end_time), rates)

    hopping_constants = [hopping_rate for i in starting_state]

    grids = [DiffEqJump.CartesianGrid1(dims), DiffEqJump.CartesianGrid2(dims), DiffEqJump.CartesianGrid3(dims), LightGraphs.grid(dims)]
    jump_probs = JumpProblem[JumpProblem(prob, NSM(), majumps, hopping_constants=hopping_constants, spatial_system = spatial_system, save_positions=(false,false)) for spatial_system in grids]
    graph = grids[end]
    num_species = 3
    hopping_constants = Vector{Matrix{Float64}}(undef, num_nodes)
    for site in 1:num_nodes
        hopping_constants[site] = hopping_rate*ones(num_species, DiffEqJump.num_neighbors(graph, site))
    end
    append!(jump_probs, JumpProblem[JumpProblem(prob, alg, majumps, hopping_constants=hopping_constants, spatial_system = graph, save_positions=(false,false)) for alg in non_spatial_algs])

    @progress "benchmarking on a $dims grid" for (i,jump_prob) in enumerate(jump_probs)
        b = @benchmarkable solve($jump_prob, SSAStepper()) samples = 15 seconds = 600
        solve_ABC_benchmarks[k][i] = run(b)
    end
end
using JLD
save("C:/Users/Vasily/Downloads/benchmarks.jld", "1D", solve_ABC_benchmarks[1], "3D", solve_ABC_benchmarks[2])