using DiffEqJump, BenchmarkTools, LightGraphs
using ProgressLogging

# benchmark solving pure diffusion
diffusivity = 0.1
linear_size = 64
dim = 3
dims = Tuple([linear_size for i in 1:dim])
num_species = 1

majumps = MassActionJump([], [], [])
tf = 0.5
tspan = (0.0, tf)
u0 = [10^4]
domain_size = 1.0 #μ-meter
num_nodes = prod(dims)

starting_state = zeros(Int, length(u0), num_nodes)
center_node = trunc(Int,(num_nodes+1)/2)
starting_state[:,center_node] = copy(u0)
prob = DiscreteProblem(starting_state,(0.0,tf), [])

hopping_rate = diffusivity * (linear_size/domain_size)^2
hopping_constants = [hopping_rate for i in starting_state]

algs = [NSM(), RDirect(), RSSA(), RSSACR()]

grids = [DiffEqJump.CartesianGrid1(dims), DiffEqJump.CartesianGrid2(dims), DiffEqJump.CartesianGrid3(dims), LightGraphs.grid(dims)]
jump_problems = JumpProblem[JumpProblem(prob, NSM(), majumps, hopping_constants=hopping_constants, spatial_system = grid, save_positions=(false,false)) for grid in grids]
# setup flattenned jump prob
graph = LightGraphs.grid(dims)
hopping_constants = Vector{Matrix{Float64}}(undef, num_nodes)
for site in 1:num_nodes
    hopping_constants[site] = hopping_rate*ones(num_species, DiffEqJump.num_neighbors(graph, site))
end
for alg in algs
    push!(jump_problems, DiffEqJump.flatten([], [], [], graph, starting_state, tspan, alg, hopping_constants, save_positions=(false,false)))
end
solve_diffusion_benchmarks = Vector{BenchmarkTools.Trial}(undef, length(jump_problems))

@progress "diffusion benchmarking on a $dims grid" for (i,spatial_jump_prob) in enumerate(jump_problems)
    b = @benchmarkable solve($spatial_jump_prob, SSAStepper())
    solve_diffusion_benchmarks[i] = run(b, samples = 15, seconds = 600)
end
