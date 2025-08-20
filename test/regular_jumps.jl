using JumpProcesses, DiffEqBase
using Test, LinearAlgebra
using StableRNGs
rng = StableRNG(12345)

Nsims = 10


β = 0.1 / 1000.0
ν = 0.01
influx_rate = 1.0
p = (β, ν, influx_rate)

function regular_rate(out, u, p, t)
    out[1] = p[1] * u[1] * u[2]
    out[2] = p[2] * u[2]
    out[3] = p[3]
end

regular_c = (dc, u, p, t, counts, mark) -> begin
    dc .= 0
    dc[1] = -counts[1] + counts[3]
    dc[2] = counts[1] - counts[2]
    dc[3] = counts[2]
end

u0 = [999, 5, 0]
tspan = (0.0, 250.0)
prob_disc = DiscreteProblem(u0, tspan, p)


rj = RegularJump(regular_rate, regular_c, 3)
jump_prob_tau = JumpProblem(prob_disc, Direct(), rj; rng=StableRNG(12345))

sol = solve(EnsembleProblem(jump_prob_tau), SimpleImplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat=1.0)
