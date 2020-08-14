using DiffEqBase, DiffEqJump
using BenchmarkTools

alg = RDirect()
counter_coeffs = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]

function time_Nsims(jprob, Nsims)
    solve(jump_prob, SSAStepper())
    res = 0
    for i in 1:Nsims
        res += @elapsed solve(jump_prob, SSAStepper())
    end
    res
end

# bimolerx
tf           = .01
u0           = [200, 100, 150]
reactstoch = [
    [1 => 2],
    [2 => 1],
    [1 => 1, 2 => 1],
    [3 => 1],
    [3 => 3]
]
netstoch = [
    [1 => -2, 2 => 1],
    [1 => 2, 2 => -1],
    [1 => -1, 2 => -1, 3 => 1],
    [1 => 1, 2 => 1, 3 => -1],
    [1 => 3, 3 => -3]
]
rates = [1., 2., .5, .75, .25]
spec_to_dep_jumps = [[1,3],[2,3],[4,5]]
jump_to_dep_specs = [[1,2],[1,2],[1,2,3],[1,2,3],[1,3]]
majumps = MassActionJump(rates, reactstoch, netstoch)
prob = DiscreteProblem(u0, (0.0, tf), rates)

println("Timing bimolerx")
for counter_coeff in counter_coeffs
    counter_threshold = trunc.(Int, counter_coeff*length(rates))
    jump_prob = JumpProblem(prob, alg, majumps, vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs, counter_threshold = counter_threshold)
    t = time_Nsims(jump_prob, 1000)
    @show counter_coeff, t
end

# extinction
reactstoch = [
    [1 => 1]
]
netstoch = [
    [1 => -1]
]
rates = [1.]
spec_to_dep_jumps = [[1]]
jump_to_dep_specs = [[1]]
dg = [[1]]
majumps = MassActionJump(rates, reactstoch, netstoch)
u0 = [10]
prob = DiscreteProblem(u0,(0.,100.),rates)

println("Timing extinction")
ts = zeros(length(counter_coeffs))
for (i,counter_coeff) in enumerate(counter_coeffs)
    counter_threshold = trunc.(Int, counter_coeff*length(rates))
    jump_prob = JumpProblem(prob, alg, majumps, vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs, counter_threshold = counter_threshold)
    t = time_Nsims(jump_prob, 1000)
    ts[i] = t
    @show counter_coeff, t
end
