using DiffEqBase, DiffEqJump, Test, StableRNGs

rng = StableRNG(12345)

# test for https://github.com/SciML/DiffEqJump.jl/issues/177
p   = [1., 2., 50.]
ns  = [[1 => 1], [1 => -1], [1 => 1]]
rs  = [[1 => 1], [1 => 1], Pair{Int64,Int64}[]]
maj = MassActionJump(p, rs, ns)
u0    = [5]
tspan = (0., 2e6)
dt    = tspan[2] / 1000
dprob = DiscreteProblem(u0, tspan, p)
jprob = JumpProblem(dprob, Direct(), maj, save_positions=(false,false), rng=rng)
sol   = solve(jprob, SSAStepper(), saveat=tspan[1]:dt:tspan[2])
@test length(unique(sol[(end-10):end][:])) > 1
