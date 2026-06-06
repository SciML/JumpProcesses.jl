using DiffEqBase, JumpProcesses, Test
using StableRNGs

rate_consts = [10.0]
reactant_stoich = [[1 => 1, 2 => 1]]
net_stoich = [[1 => -1, 2 => -1, 3 => 1]]
maj = MassActionJump(rate_consts, reactant_stoich, net_stoich)

n0 = [1, 1, 0]
tspan = (0, 0.2)
dprob = DiscreteProblem(n0, tspan)
ts = collect(0:0.002:tspan[2])
Nsims = 10_000

# Give every trajectory its own independent, seeded RNG. EnsembleProblems default to
# EnsembleThreads(), so sharing one mutable RNG across trajectories is a data race that
# corrupts the RNG state. A fresh StableRNG per trajectory keeps the solve race-free and
# reproducible.
function jprob_func(save_positions)
    function (prob, ctx)
        rng = StableRNG(12345 + ctx.sim_id)
        JumpProblem(dprob, Direct(), maj; save_positions, rng)
    end
end

jprob = JumpProblem(dprob, Direct(), maj, save_positions = (false, false),
    rng = StableRNG(12345))
NA = zeros(length(ts))
sol = JumpProcesses.solve(
    EnsembleProblem(jprob; prob_func = jprob_func((false, false))), SSAStepper(),
    saveat = ts, trajectories = Nsims)

for i in 1:length(sol.u)
    NA .+= sol.u[i][1, :]
end

for i in 1:length(ts)
    @test NA[i] / Nsims≈exp(-10 * ts[i]) rtol=1e-1
end

NA = zeros(length(ts))
jprob = JumpProblem(dprob, Direct(), maj; rng = StableRNG(12345))
sol = nothing;
GC.gc();
sol = JumpProcesses.solve(
    EnsembleProblem(jprob; prob_func = jprob_func((false, true))), SSAStepper(),
    trajectories = Nsims)

for i in 1:Nsims
    for n in 1:length(ts)
        NA[n] += sol.u[i](ts[n])[1]
    end
end

for i in 1:length(ts)
    @test NA[i] / Nsims≈exp(-10 * ts[i]) rtol=1e-1
end
