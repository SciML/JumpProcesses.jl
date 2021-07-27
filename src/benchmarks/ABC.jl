# benchmarks for various grids
using DiffEqJump, BenchmarkTools, LightGraphs
using ProgressLogging

linear_size = 32
dim = 3
dims = Tuple([linear_size for i in 1:dim])
starting_site = trunc(Int,(prod(dims) + 1)/2)
u0 = [linear_size^dim,linear_size^dim,0]
end_time = 0.01
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

non_spatial_algs = [NRM(), RSSACR(), DirectCR()]
grids = [DiffEqJump.CartesianGrid1(dims), DiffEqJump.CartesianGrid2(dims), DiffEqJump.CartesianGrid3(dims), LightGraphs.grid(dims)]
jump_probs = JumpProblem[JumpProblem(prob, NSM(), majumps, hopping_constants=hopping_constants, spatial_system = spatial_system, save_positions=(false,false)) for spatial_system in grids]
append!(jump_probs, JumpProblem[JumpProblem(prob, alg, majumps, hopping_constants=hopping_constants, spatial_system = spatial_system, save_positions=(false,false)) for alg in non_spatial_algs])

solve_ABC_benchmarks = Vector{BenchmarkTools.Trial}(undef, length(jump_probs))

for (i,grid) in enumerate(grids)
    spatial_jump_prob = JumpProblem(prob, NSM(), majumps, hopping_constants=hopping_constants, spatial_system = spatial_system, save_positions=(false,false))
    b = @benchmarkable solve($spatial_jump_prob, SSAStepper()) samples = 15 seconds = 600
    solve_ABC_benchmarks[i] = run(b)
end
