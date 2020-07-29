# Set up a spatial gene expression model and solve it
using DiffEqJump, DiffEqBase

tf           = 1000.0
u0           = [1,0,0,0]

reactstoch = [
    [1 => 1],
    [2 => 1],
    [2 => 1],
    [3 => 1],
    [1 => 1, 3 => 1],
    [4 => 1]
]
netstoch = [
    [2 => 1],
    [3 => 1],
    [2 => -1],
    [3 => -1],
    [1 => -1, 3 => -1, 4 => 1],
    [1 => 1, 3 => 1, 4 => -1]
]
spec_to_dep_jumps = [[1,5],[2,3],[4,5],[6]]
jump_to_dep_specs = [[2],[3],[2],[3],[1,3,4],[1,3,4]]
rates = [.5, (20*log(2.)/120.), (log(2.)/120.), (log(2.)/600.), .025, 1.]
majumps = MassActionJump(rates, reactstoch, netstoch)
prob = DiscreteProblem(u0, (0.0, tf), rates)

# Graph setup for gene expression model
num_nodes = 30
connectivity_list = [[mod1(i-1,num_nodes),mod1(i+1,num_nodes)] for i in 1:num_nodes] # this is a cycle graph

diff_rates_for_edge = Array{Float64,1}(undef,length(jump_prob_gene_expr.prob.u0))
diff_rates_for_edge[1] = 0.01
diff_rates_for_edge[2] = 0.01
diff_rates_for_edge[3] = 1.0
diff_rates_for_edge[4] = 1.0

starting_state = zeros(Int, num_nodes*length(prob.u0))
starting_state[1 : length(prob.u0)] = copy(prob.u0)
alg = WellMixedSpatial(NRM())
spatial_jump_prob = JumpProblem(prob, alg, majumps; connectivity_list = connectivity_list, diff_rates = diff_rates_for_edge, starting_state = starting_state, save_positions=(false,false))
sol = solve(spatial_jump_prob, SSAStepper(), saveat = prob.tspan[2]/200)
