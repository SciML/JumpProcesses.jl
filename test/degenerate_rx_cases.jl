# testing some special cases that can get missed in more complicated tests
# these are more checks for crashes than accuracy
# tests zeroth order mass action reactions and passing just one mass action reaction
# tests FW and tuple based mix of mass action jump / constant jump

using DiffEqBase, JumpProcesses
using Test
using StableRNGs
rng = StableRNG(12345)

doprint = false
#using Plots; plotlyjs()
doplot = false

methods = (JumpProcesses.JUMP_AGGREGATORS..., JumpProcesses.NullAggregator)

# one reaction case, mass action jump, vector of data
rate = [2.0]
rs = [[0 => 1]]
ns = [[1 => 1]]
jump = MassActionJump(rate, rs, ns)
prob = DiscreteProblem([100], (0.0, 100.0))
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
sol = solve(jump_prob, SSAStepper())
if doprint
    println("mass act jump using vectors of data: last val = ", sol[end, end])
end
if doplot
    plothand = plot(sol, lab = "vec, 0=>1")
end
@test sol[end, end] > 200

# one reaction case, mass action jump, data as scalars
rate = 2.0
rs = [0 => 3]                   # stoich power should be ignored
ns = [1 => 1]
jump = MassActionJump(rate, rs, ns)
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
sol = solve(jump_prob, SSAStepper())
if doprint
    println("mass act jump using scalar data: last val = ", sol[end, end])
end
if doplot
    plot!(plothand, sol, lab = "scalar, 0=>1")
end
@test sol[end, end] > 200

# one reaction case, mass action jump, vector of empty data
rate = [2.0]
rs = [Vector{Pair{Int, Int}}()]
ns = [[1 => 1]]
jump = MassActionJump(rate, rs, ns)
prob = DiscreteProblem([100], (0.0, 100.0))
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
sol = solve(jump_prob, SSAStepper())
if doprint
    println("mass act jump using vector of Pair{Int,Int}: last val = ", sol[end, end])
end
if doplot
    plot!(plothand, sol, lab = "vec, ()")
end
@test sol[end, end] > 200

# one reaction case, mass action jump, data as scalars
rate = 2.0
rs = Vector{Pair{Int, Int}}()
ns = [1 => 1]
jump = MassActionJump(rate, rs, ns)
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
sol = solve(jump_prob, SSAStepper())
if doprint
    println("mass act jump using scalar Pair{Int,Int}: last val = ", sol[end, end])
end
if doplot
    plot!(plothand, sol, lab = "scalar, ()")
end
@test sol[end, end] > 200

# mix two rx types
# 0 -> A and A -> 0
rate = 600.0
rs = [0 => 3]                   # stoich power should be ignored
ns = [1 => 1]
jump = MassActionJump(rate, rs, ns)
ratefun = (u, p, t) -> 2.0 * u[1]
affect! = function (integrator)
    integrator.u[1] -= 1
end
jump2 = ConstantRateJump(ratefun, affect!)
if doplot
    plothand2 = plot(reuse = false)
end

dep_graph = [
    [1, 2],
    [1, 2]
]
spec_to_dep_jumps = [[2]]
jump_to_dep_specs = [[1], [1]]
namedpars = (dep_graph = dep_graph, vartojumps_map = spec_to_dep_jumps,
    jumptovars_map = jump_to_dep_specs)

for method in methods
    local jump_prob = JumpProblem(prob, method, jump, jump2; rng = rng, namedpars...)
    local sol = solve(jump_prob, SSAStepper())

    if doplot
        plot!(plothand2, sol, label = ("A <-> 0, " * string(method)))
    end

    if doprint
        println("Mix of constant and mass action jumps, method = ", typeof(method),
            ", sol[end] = ", sol[end, end])
    end
    @test sol[end, end] > 200
end

if doplot
    display(plothand)
    display(plothand2)
end

# check MassActionJump with parameters merge together correctly
rs1 = [1 => 1]
rs2 = [2 => 1]
ns1 = [1 => -1, 2 => 1]
ns2 = [1 => 1, 2 => -1]
p = [1.0, 0.0]
maj1 = MassActionJump(rs1, ns1; param_idxs = 1)
maj2 = MassActionJump(rs2, ns2; param_idxs = 2)
js = JumpSet(maj1, maj2)
maj = MassActionJump([rs1, rs2], [ns1, ns2]; param_idxs = [1, 2])
@test all(getfield(maj, fn) == getfield(js.massaction_jump, fn)
for fn in [:scaled_rates, :reactant_stoch, :net_stoch])
@test all(maj.param_mapper.param_idxs .== js.massaction_jump.param_mapper.param_idxs)
maj1 = MassActionJump([rs1], [ns1]; param_idxs = [1])
maj2 = MassActionJump([rs2], [ns2]; param_idxs = [2])
js = JumpSet(maj1, maj2)
@test all(getfield(maj, fn) == getfield(js.massaction_jump, fn)
for fn in [:scaled_rates, :reactant_stoch, :net_stoch])
@test all(maj.param_mapper.param_idxs .== js.massaction_jump.param_mapper.param_idxs)
