using DiffEqJump, DiffEqBase
using Test
using LinearAlgebra
using LightGraphs

function get_mean_sol(jump_prob, Nsims, saveat)
    sol = solve(jump_prob, SSAStepper(), saveat = saveat).u
    for i in 1:Nsims-1
        sol += solve(jump_prob, SSAStepper(), saveat = saveat).u
    end
    sol/Nsims
end

# assume sites are labeled from 1 to num_sites(spatial_system)
function discrete_laplacian_from_spatial_system(spatial_system, hopping_rate)
    sites = 1:DiffEqJump.num_sites(spatial_system)
    laplacian = zeros(Int, length(sites), length(sites))
    for site in sites
        laplacian[site,site] = -DiffEqJump.num_neighbors(spatial_system, site)
        for nb in DiffEqJump.neighbors(spatial_system, site)
            laplacian[site, nb] = 1
        end
    end
    laplacian*hopping_rate # is this part of laplacian?
end

reactstoch = []
netstoch = []
rates = []
majumps = MassActionJump(rates, reactstoch, netstoch)
tf = 0.5
u0 = [100]

domain_size = 1.0 #μ-meter
linear_size = 5
diffusivity = 0.1
dim = 2
grid = LightGraphs.grid([linear_size for i in 1:dim])
num_nodes = DiffEqJump.num_sites(grid)

# Starting state setup
starting_state = zeros(Int, length(u0), num_nodes)
center_node = trunc(Int,(num_nodes+1)/2)
starting_state[:,center_node] = copy(u0)
prob = DiscreteProblem(starting_state,(0.0,tf), rates)

hopping_rate = diffusivity * (linear_size/domain_size)^2
hopping_constants = [hopping_rate for i in starting_state]

alg = NSM()
spatial_jump_prob = JumpProblem(prob, alg, majumps, hopping_constants=hopping_constants, spatial_system=grid, save_positions=(false,false))

num_time_points = 10
Nsims = 10000
solution = solve(spatial_jump_prob, SSAStepper(), saveat = tf/num_time_points).u
mean_sol = get_mean_sol(spatial_jump_prob, Nsims, tf/num_time_points)

# analytic solution
lap = discrete_laplacian_from_spatial_system(grid, hopping_rate)
evals, B = eigen(lap) # lap == B*diagm(evals)*B'
Bt = B'
analytic_solution(t) = B*diagm(ℯ.^(t*evals))*Bt * reshape(prob.u0, num_nodes, 1)

# test mean solution
# n = num_nodes
# D = diffusivity
# h = domain_size/linear_size
# eval_lap(j, h, n) = -4/h^2 * (sin(pi*(j-1)/(2n)))^2
# evec_lap(j,n) = j==1 ? [√(1/n) for i in 1:n] : [√(2/n) * cos(pi*(j-1)*(i-0.5)/n) for i in 1:n]
# Q = hcat([evec_lap(j,n) for j in 1:n]...)
# Qt = Q'
# lambdas = [eval_lap(j, h, n) for j in 1:n]
# analytic_solution2(t) = Q*diagm(ℯ.^(D*t*lambdas))*Qt * reshape(prob.u0, num_nodes, 1)

rel_tol = 0.01

times = 0.0:tf/num_time_points:tf
for (i,t) in enumerate(times)
    local diff = analytic_solution(t) - reshape(mean_sol[i], num_nodes, 1)
    @test abs(sum(diff[1:center_node])/sum(analytic_solution(t)[1:center_node])) < rel_tol
end
