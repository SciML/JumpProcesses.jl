using DiffEqJump, DiffEqBase
# using BenchmarkTools
using Test, Graphs

dim = 1
linear_size = 5
dims = Tuple(repeat([linear_size], dim))
num_nodes = prod(dims)
starting_site = trunc(Int,(linear_size^dim + 1)/2)
u0 = [0]
end_time = 10.0
diffusivity = 1.0

num_species = 1
reactstoch = [[1 => 1]]
netstoch = [[1 => 1],[1 => -1]]
uniform_rates = ones(2, num_nodes)
uniform_rates[2,:] *= 0.01
non_uniform_rates = zeros(2, num_nodes)
non_uniform_rates[:,starting_site] = uniform_rates[:,starting_site]

# majumps
uniform_majumps_1 = SpatialMassActionJump(uniform_rates[:,1], reactstoch, netstoch)
uniform_majumps_2 = SpatialMassActionJump(uniform_rates, reactstoch, netstoch)
uniform_majumps_3 = SpatialMassActionJump([1.], reshape(uniform_rates[2,:], 1, num_nodes), reactstoch, netstoch) # hybrid
uniform_majumps_4 = SpatialMassActionJump(MassActionJump(uniform_rates[:,1], reactstoch, netstoch))
uniform_majumps = [uniform_majumps_1, uniform_majumps_2, uniform_majumps_3, uniform_majumps_4]

non_uniform_majumps_1 = SpatialMassActionJump(non_uniform_rates, reactstoch, netstoch) # reactions are zero outside of starting state
non_uniform_majumps_2 = SpatialMassActionJump([1.], reshape(non_uniform_rates[2,:], 1, num_nodes), non_uniform_rates, reactstoch,netstoch) # birth everywhere, death only at starting state
non_uniform_majumps_3 = SpatialMassActionJump([1. 0. 0. 0. 0.; 0. 0. 0. 0. 1.], reactstoch,netstoch) # birth on the left, death on the right
non_uniform_majumps = [non_uniform_majumps_1, non_uniform_majumps_2, non_uniform_majumps_3]


# DiscreteProblem setup
starting_state = zeros(Int, length(u0), num_nodes)
tspan = (0.0, end_time)
prob = DiscreteProblem(starting_state, tspan, rates)
hopping_constants = [diffusivity for i in starting_state]



function get_mean_end_state(jump_prob, Nsims)
    end_state = zeros(size(jump_prob.prob.u0))
    for i in 1:Nsims
        sol = solve(jump_prob, SSAStepper())
        end_state .+= sol.u[end]
    end
    end_state/Nsims
end

# testing
grid = Graphs.grid(dims)
uniform_jump_problems = JumpProblem[JumpProblem(prob, NSM(), majump, hopping_constants=hopping_constants, spatial_system = grid, save_positions=(false,false)) for majump in uniform_majumps]
# append flattenned jump probs
append!(uniform_jump_problems, JumpProblem[JumpProblem(prob, NRM(), majump, hopping_constants=hopping_constants, spatial_system = grid, save_positions=(false,false)) for majump in uniform_majumps])

# TODO write the accuracy tests
# test
for spatial_jump_prob in jump_problems
    solution = solve(spatial_jump_prob, SSAStepper())
    mean_end_state = get_mean_end_state(spatial_jump_prob, Nsims)
    mean_end_state = reshape(mean_end_state, num_species, num_nodes)
    diff =  sum(mean_end_state, dims = 2) - non_spatial_mean
    for (i,d) in enumerate(diff)
        @test abs(d) < reltol*non_spatial_mean[i]
    end
end

# non-uniform
non_uniform_jump_problems = JumpProblem[JumpProblem(prob, NSM(), majump, hopping_constants=hopping_constants, spatial_system = grid, save_positions=(false,false)) for majump in non_uniform_majumps]

#using non-spatial SSAs to get the mean
# non_spatial_rates = [0.1,1.0]
# reactstoch = [[1 => 1, 2 => 1],[3 => 1]]
# netstoch = [[1 => -1, 2 => -1, 3 => 1],[1 => 1, 2 => 1, 3 => -1]]
# majumps = MassActionJump(non_spatial_rates, reactstoch, netstoch)
# non_spatial_prob = DiscreteProblem(u0,(0.0,end_time), non_spatial_rates)
# jump_prob = JumpProblem(non_spatial_prob, Direct(), majumps)
# non_spatial_mean = get_mean_end_state(jump_prob, 10000)