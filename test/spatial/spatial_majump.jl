using JumpProcesses, DiffEqBase, OrdinaryDiffEq
using Test, Graphs, LinearAlgebra
using StableRNGs
rng = StableRNG(12345)

reltol = 0.05
Nsims = 10^3

dim = 1
linear_size = 5
dims = Tuple(repeat([linear_size], dim))
num_nodes = prod(dims)
center_site = trunc(Int, (linear_size^dim + 1) / 2)
u0 = zeros(Int, 1, num_nodes)
end_time = 100.0
diffusivity = 1.0
death_rate = 0.1

num_species = 1
reactstoch = [Pair{Int64, Int64}[], [1 => 1]]
netstoch = [[1 => 1], [1 => -1]]
uniform_rates = ones(2, num_nodes)
uniform_rates[2, :] *= death_rate
non_uniform_rates = zeros(2, num_nodes)
non_uniform_rates[:, center_site] = uniform_rates[:, center_site]

# DiscreteProblem setup
tspan = (0.0, end_time)
prob = DiscreteProblem(u0, tspan, uniform_rates)

# spatial system
grid = Graphs.grid(dims)
hopping_constants = [diffusivity for i in u0]

# majumps
uniform_majumps_1 = SpatialMassActionJump(uniform_rates[:, 1], reactstoch, netstoch)
uniform_majumps_2 = SpatialMassActionJump(uniform_rates, reactstoch, netstoch)
uniform_majumps_3 = SpatialMassActionJump([1.0], reshape(uniform_rates[2, :], 1, num_nodes),
                                          reactstoch, netstoch) # hybrid
uniform_majumps_4 = SpatialMassActionJump(MassActionJump(uniform_rates[:, 1], reactstoch,
                                                         netstoch))
uniform_majumps = [
    uniform_majumps_1,
    uniform_majumps_2,
    uniform_majumps_3,
    uniform_majumps_4,
]

non_uniform_majumps_1 = SpatialMassActionJump(non_uniform_rates, reactstoch, netstoch) # reactions are zero outside of center site
non_uniform_majumps_2 = SpatialMassActionJump([1.0],
                                              reshape(non_uniform_rates[2, :], 1,
                                                      num_nodes), reactstoch, netstoch) # birth everywhere, death only at center site
non_uniform_majumps_3 = SpatialMassActionJump([1.0 0.0 0.0 0.0 0.0;
                                               0.0 0.0 0.0 0.0 death_rate], reactstoch,
                                              netstoch) # birth on the left, death on the right
non_uniform_majumps = [non_uniform_majumps_1, non_uniform_majumps_2, non_uniform_majumps_3]

# put together the JumpProblem's
uniform_jump_problems = JumpProblem[JumpProblem(prob, NSM(), majump,
                                                hopping_constants = hopping_constants,
                                                spatial_system = grid,
                                                save_positions = (false, false), rng = rng)
                                    for majump in uniform_majumps]
# flattenned
append!(uniform_jump_problems,
        JumpProblem[JumpProblem(prob, NRM(), majump, hopping_constants = hopping_constants,
                                spatial_system = grid, save_positions = (false, false), rng = rng)
                    for majump in uniform_majumps])

# non-uniform
non_uniform_jump_problems = JumpProblem[JumpProblem(prob, NSM(), majump,
                                                    hopping_constants = hopping_constants,
                                                    spatial_system = grid,
                                                    save_positions = (false, false), rng = rng)
                                        for majump in non_uniform_majumps]

# testing
function get_mean_end_state(jump_prob, Nsims)
    end_state = zeros(size(jump_prob.prob.u0))
    for i in 1:Nsims
        sol = solve(jump_prob, SSAStepper())
        end_state .+= sol.u[end]
    end
    end_state / Nsims
end

function discrete_laplacian_from_spatial_system(spatial_system, hopping_rate)
    sites = 1:num_sites(spatial_system)
    laplacian = zeros(length(sites), length(sites))
    for site in sites
        laplacian[site, site] = -outdegree(spatial_system, site)
        for nb in neighbors(spatial_system, site)
            laplacian[site, nb] = 1
        end
    end
    laplacian .*= hopping_rate
    laplacian
end
L = discrete_laplacian_from_spatial_system(grid, diffusivity)

# birth and death everywhere
f(u, p, t) = L * u - death_rate * u + uniform_rates[1, :]
ode_prob = ODEProblem(f, zeros(num_nodes), tspan)
sol = solve(ode_prob, Tsit5())

for spatial_jump_prob in uniform_jump_problems
    solution = solve(spatial_jump_prob, SSAStepper())
    mean_end_state = get_mean_end_state(spatial_jump_prob, Nsims)
    mean_end_state = reshape(mean_end_state, num_nodes)
    diff = mean_end_state - sol.u[end]
    for (i, d) in enumerate(diff)
        @test abs(d) < reltol * sol.u[end][i]
    end
end

# birth and death zero outside of center site
f(u, p, t) = L * u - diagm([0.0, 0.0, death_rate, 0.0, 0.0]) * u + [0.0, 0.0, 1.0, 0.0, 0.0]
ode_prob = ODEProblem(f, zeros(num_nodes), tspan)
sol = solve(ode_prob, Tsit5())

solution = solve(non_uniform_jump_problems[1], SSAStepper())
mean_end_state = get_mean_end_state(non_uniform_jump_problems[1], Nsims)
mean_end_state = reshape(mean_end_state, num_nodes)
diff = mean_end_state - sol.u[end]
for (i, d) in enumerate(diff)
    @test abs(d) < reltol * sol.u[end][i]
end

# birth everywhere, death only at center site
f(u, p, t) = L * u - diagm([0.0, 0.0, death_rate, 0.0, 0.0]) * u + ones(num_nodes)
ode_prob = ODEProblem(f, zeros(num_nodes), tspan)
sol = solve(ode_prob, Tsit5())

solution = solve(non_uniform_jump_problems[2], SSAStepper())
mean_end_state = get_mean_end_state(non_uniform_jump_problems[2], Nsims)
mean_end_state = reshape(mean_end_state, num_nodes)
diff = mean_end_state - sol.u[end]
for (i, d) in enumerate(diff)
    @test abs(d) < reltol * sol.u[end][i]
end

# birth on left end, death on right end
f(u, p, t) = L * u - diagm([0.0, 0.0, 0.0, 0.0, death_rate]) * u + [1.0, 0.0, 0.0, 0.0, 0.0]
ode_prob = ODEProblem(f, zeros(num_nodes), tspan)
sol = solve(ode_prob, Tsit5())

solution = solve(non_uniform_jump_problems[3], SSAStepper())
mean_end_state = get_mean_end_state(non_uniform_jump_problems[3], Nsims)
mean_end_state = reshape(mean_end_state, num_nodes)
diff = mean_end_state - sol.u[end]
for (i, d) in enumerate(diff)
    @test abs(d) < reltol * sol.u[end][i]
end
