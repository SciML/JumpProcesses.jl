using DiffEqBase, DiffEqJump, Test
using StableRNGs
rng = StableRNG(12345)

rate_consts = [10.0]
reactant_stoich = [[1 => 1, 2 => 1]]
net_stoich = [[1 => -1, 2 => -1, 3 => 1]]
maj = MassActionJump(rate_consts, reactant_stoich, net_stoich)

n0 = [1,1,0]
tspan = (0,.2)
dprob = DiscreteProblem(n0, tspan)
jprob = JumpProblem(dprob, Direct(), maj, save_positions=(false,false), rng=rng)
ts = collect(0:.002:tspan[2])
NA = zeros(length(ts))
Nsims = 10_000
sol = DiffEqJump.solve(EnsembleProblem(jprob), SSAStepper(), saveat=ts, trajectories=Nsims)

for i in 1:length(sol)
    NA .+= sol[i][1,:]
end

for i in 1:length(ts)
    @test NA[i] / Nsims ≈ exp(-10*ts[i]) rtol=1e-1
end

NA = zeros(length(ts))
jprob = JumpProblem(dprob, Direct(), maj; rng=rng)
sol = nothing; GC.gc()
sol = DiffEqJump.solve(EnsembleProblem(jprob), SSAStepper(), trajectories=Nsims)

for i = 1:Nsims
    for n = 1:length(ts)
        NA[n] += sol[i](ts[n])[1]
    end
end

for i in 1:length(ts)
    @test NA[i] / Nsims ≈ exp(-10*ts[i]) rtol=1e-1
end
