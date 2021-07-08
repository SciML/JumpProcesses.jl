using DiffEqJump, DiffEqBase
# using BenchmarkTools
using Test, LightGraphs

function ABC_setup(spatial_system, starting_site, u0, diffusivity, end_time)
    domain_size = 1.0 #μ-meter
    mesh_size = domain_size/linear_size
    # ABC model A + B <--> C
    reactstoch = [[1 => 1, 2 => 1],[3 => 1]]
    netstoch = [[1 => -1, 2 => -1, 3 => 1],[1 => 1, 2 => 1, 3 => -1]]
    rates = [0.1/mesh_size, 1.]
    majumps = MassActionJump(rates, reactstoch, netstoch)
    
    # spatial system setup
    hopping_rate = diffusivity * (linear_size/domain_size)^2
    num_nodes = DiffEqJump.num_sites(spatial_system)
    
    # Starting state setup
    starting_state = zeros(Int, length(u0), num_nodes)
    starting_state[:,starting_site] = copy(u0)
    prob = DiscreteProblem(starting_state,(0.0,end_time), rates)
    
    hopping_constants = [hopping_rate for i in starting_state]
    
    alg = NSM()
    return JumpProblem(prob, alg, majumps, hopping_constants=hopping_constants, spatial_system = spatial_system, save_positions=(false,false))
end

function get_mean_end_state(jump_prob, Nsims)
    end_state = zeros(size(jump_prob.prob.u0))
    for i in 1:Nsims
        sol = solve(jump_prob, SSAStepper())
        end_state += sol.u[end]
    end
    end_state/Nsims
end

Nsims        = 100
reltol       = 0.05
non_spatial_mean = [65.7395, 65.7395, 434.2605] #mean of 10,000 simulations

dim = 1
linear_size = 5
starting_site = trunc(Int,(linear_size^dim + 1)/2)
u0 = [500,500,0]
end_time = 10.0
diffusivity = 1.0

# testing on CartesianGrid
grid = CartesianGrid(dim, linear_size)
spatial_jump_prob = ABC_setup(grid, starting_site, u0, diffusivity, end_time)
solution = solve(spatial_jump_prob, SSAStepper())
mean_end_state = get_mean_end_state(spatial_jump_prob, Nsims)
diff =  sum(mean_end_state, dims = 2) - non_spatial_mean
# println("max relative error: $(maximum(abs.(diff./non_spatial_mean)))")
for (i,d) in enumerate(diff)
    @test abs(d) < reltol*non_spatial_mean[i]
end

# testing on LightGraphs
lgrid = LightGraphs.grid([linear_size])
lspatial_jump_prob = ABC_setup(lgrid, starting_site, u0, diffusivity, end_time)
lsolution = solve(lspatial_jump_prob, SSAStepper())

lmean_end_state = get_mean_end_state(lspatial_jump_prob, Nsims)

diff =  sum(lmean_end_state, dims = 2) - non_spatial_mean
# println("max relative error: $(maximum(abs.(diff./non_spatial_mean)))")
for (i,d) in enumerate(diff)
    @test abs(d) < reltol*non_spatial_mean[i]
end



#using non-spatial SSAs to get the mean
# non_spatial_rates = [0.1,1.0]
# reactstoch = [[1 => 1, 2 => 1],[3 => 1]]
# netstoch = [[1 => -1, 2 => -1, 3 => 1],[1 => 1, 2 => 1, 3 => -1]]
# majumps = MassActionJump(non_spatial_rates, reactstoch, netstoch)
# non_spatial_prob = DiscreteProblem(u0,(0.0,end_time), non_spatial_rates)
# jump_prob = JumpProblem(non_spatial_prob, Direct(), majumps)
# non_spatial_mean = get_mean_end_state(jump_prob, 10000)