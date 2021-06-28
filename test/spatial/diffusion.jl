using DiffEqJump, DiffEqBase
using Test
using LinearAlgebra

function get_mean_sol(jump_prob, Nsims, saveat)
    sol = solve(jump_prob, SSAStepper(), saveat = saveat).u
    for i in 1:Nsims-1
        sol += solve(jump_prob, SSAStepper(), saveat = saveat).u
    end
    sol/Nsims
end

reactstoch = []
netstoch = []
rates = []
majumps = MassActionJump(rates, reactstoch, netstoch)
tf = 5.0
u0 = [100]

domain_size = 1.0 #μ-meter
num_sites_per_edge = 32
diffusivity = 0.1
hopping_rate = diffusivity * (num_sites_per_edge/domain_size)^2
dimension = 1
connectivity_list = connectivity_list_from_box(num_sites_per_edge, dimension) # this is a grid graph
num_nodes = length(connectivity_list)

# Starting state setup
starting_state = zeros(Int, num_nodes*length(u0))
center_node = coordinates_to_node(trunc(Int,(num_sites_per_edge+1)/2),num_sites_per_edge)
center_node_first_species_index = to_spatial_spec(center_node, 1, length(u0))
starting_state[center_node_first_species_index : center_node_first_species_index + length(u0) - 1] = copy(u0)
prob = DiscreteProblem(starting_state,(0.0,tf), rates)

alg = WellMixedSpatial(RSSACR())
spatial_jump_prob = JumpProblem(prob, alg, majumps; connectivity_list = connectivity_list, diff_rates = hopping_rate)

num_time_points = 10
Nsims = 1000
mean_sol = get_mean_sol(spatial_jump_prob, Nsims, tf/num_time_points)

# test mean solution
n = num_nodes
h = domain_size/num_sites_per_edge
D = diffusivity
eval_lap(j, h, n) = -4/h^2 * (sin(pi*(j-1)/(2n)))^2
evec_lap(j,n) = j==1 ? [√(1/n) for i in 1:n] : [√(2/n) * cos(pi*(j-1)*(i-0.5)/n) for i in 1:n]
Q = hcat([evec_lap(j,n) for j in 1:n]...)
Qt = Q'
lambdas = [eval_lap(j, h, n) for j in 1:n]
analytic_solution(t) = Q*diagm(ℯ.^(D*t*lambdas))*Qt * prob.u0

rel_tol = 0.01

for (i,t) in enumerate(0.0:tf/num_time_points:tf)
    diff = analytic_solution(t) - mean_sol[i]
    center = Int(num_sites_per_edge/2)
    @test sum(diff[1:center])/sum(analytic_solution(t)[1:center]) < rel_tol
end
