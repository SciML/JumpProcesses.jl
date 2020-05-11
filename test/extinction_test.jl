using DiffEqBase, DiffEqJump, StaticArrays
using Test

reactstoch =
[
    [1 => 1]
]

netstoch =
[
    [1 => -1]
]

rates = [1.]
u0 = [10]
spec_to_dep_jumps = [[1]]
jump_to_dep_specs = [[1]]
dg = [[1]]
majump = MassActionJump(rates, reactstoch, netstoch)
dprob = DiscreteProblem(u0,(0.,10.),rates)


algs = DiffEqJump.JUMP_AGGREGATORS

for ssa in algs
    jprob = JumpProblem(dprob, ssa, majump; vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
    sol = solve(jprob, SSAStepper())
    @test sol[1,end] == 0
    @test sol.t[end] < Inf
end

u0 = SA[10]
dprob = DiscreteProblem(u0,(0.,10.),rates)

for ssa in algs
    jprob = JumpProblem(dprob, ssa, majump; vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)
    sol = solve(jprob, SSAStepper())
    @test sol[1,end] == 0
    @test sol.t[end] < Inf
end
