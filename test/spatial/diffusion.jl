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
linear_size = 32
diffusivity = 0.1
dim = 1
grid = DiffEqJump.CartesianGrid(dim, linear_size)
num_nodes = DiffEqJump.num_sites(grid)

# Starting state setup
starting_state = zeros(Int, length(u0), num_nodes)
center_node = trunc(Int,(num_nodes+1)/2)
starting_state[:,center_node] = copy(u0)
prob = DiscreteProblem(starting_state,(0.0,tf), rates)

hopping_rate = diffusivity * (linear_size/domain_size)^2
diffusion_constants = [hopping_rate for i in starting_state]

alg = NSM()
spatial_jump_prob = JumpProblem(prob, alg, majumps, diffusion_constants=diffusion_constants, spatial_system=grid, save_positions=(false,false))

num_time_points = 10
Nsims = 1000
solution = solve(spatial_jump_prob, SSAStepper(), saveat = tf/num_time_points).u
mean_sol = get_mean_sol(spatial_jump_prob, Nsims, tf/num_time_points)

# test mean solution
n = num_nodes
h = domain_size/linear_size
D = diffusivity
eval_lap(j, h, n) = -4/h^2 * (sin(pi*(j-1)/(2n)))^2
evec_lap(j,n) = j==1 ? [√(1/n) for i in 1:n] : [√(2/n) * cos(pi*(j-1)*(i-0.5)/n) for i in 1:n]
Q = hcat([evec_lap(j,n) for j in 1:n]...)
Qt = Q'
lambdas = [eval_lap(j, h, n) for j in 1:n]
analytic_solution(t) = Q*diagm(ℯ.^(D*t*lambdas))*Qt * reshape(prob.u0, num_nodes, 1)

rel_tol = 0.01

for (i,t) in enumerate(0.0:tf/num_time_points:tf)
    diff = analytic_solution(t) - reshape(mean_sol[i], num_nodes, 1)
    center = Int(linear_size/2)
    @test abs(sum(diff[1:center])/sum(analytic_solution(t)[1:center])) < rel_tol
end
