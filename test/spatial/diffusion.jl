using DiffEqJump, DiffEqBase
using Test

function get_mean_sol(jump_prob, Nsims, saveat)
    sol = solve(jump_prob, SSAStepper(), saveat = saveat).u
    for i in 1:Nsims-1
        sol += solve(jump_prob, SSAStepper(), saveat = saveat).u
    end
    sol
end

reactstoch = []
netstoch = []
rates = []
majumps = MassActionJump(rates, reactstoch, netstoch)
tf = 10.0
u0 = [1]

domain_size = 1.0 #μ-meter
num_sites_per_edge = 256
diffusivity = 0.01
hopping_rate = diffusivity * (num_sites_per_edge/domain_size)^2
dimension = 1
connectivity_list = connectivity_list_from_box(num_sites_per_edge, dimension) # this is a grid graph
num_nodes = length(connectivity_list)

# Starting state setup
starting_state = zeros(Int, num_nodes*length(u0))
center_node = coordinates_to_node(trunc(Int,num_sites_per_edge/2),num_sites_per_edge)
center_node_first_species_index = to_spatial_spec(center_node, 1, length(u0))
starting_state[center_node_first_species_index : center_node_first_species_index + length(u0) - 1] = copy(u0)
prob = DiscreteProblem(starting_state,(0.0,tf), rates)

alg = WellMixedSpatial(RSSACR())
spatial_jump_prob = JumpProblem(prob, alg, majumps; connectivity_list = connectivity_list, diff_rates = hopping_rate)
mean_sol = get_mean_sol(spatial_jump_prob, 1000, tf/100)

# test mean solution
# TODO: get an analytic solution via the discrete laplacian
