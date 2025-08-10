using JumpProcesses, DiffEqBase
using Test, LinearAlgebra
using StableRNGs, Plots
rng = StableRNG(12345)

# Parameters
c1 = 1.0      # S1 -> 0
c2 = 10.0     # S1 + S1 <- S2
c3 = 1000.0   # S1 + S1 -> S2
c4 = 0.1      # S2 -> S3
p = (c1, c2, c3, c4)

regular_rate_implicit = (out, u, p, t) -> begin
    out[1] = p[1] * u[1]          # S1 -> 0
    out[2] = p[2] * u[2]          # S1 + S1 <- S2
    out[3] = p[3] * u[1] * (u[1] - 1) / 2  # S1 + S1 -> S2
    out[4] = p[4] * u[2]          # S2 -> S3
end

regular_c_implicit = (dc, u, p, t, counts, mark) -> begin
    dc .= 0.0
    dc[1] = -counts[1] - 2 * counts[3] + 2 * counts[2]  # S1: -decay - 2*forward + 2*backward
    dc[2] = counts[3] - counts[2] - counts[4]           # S2: +forward - backward - decay
    dc[3] = counts[4]                                   # S3: +decay
end

u0 = [10000.0, 0.0, 0.0]  # S1, S2, S3
tspan = (0.0, 4.0)

# Create JumpProblem with proper parameter passing
prob_disc = DiscreteProblem(u0, tspan, p)
rj = RegularJump(regular_rate_implicit, regular_c_implicit, 4)
jump_prob = JumpProblem(prob_disc, Direct(), rj; rng=StableRNG(12345))
sol = solve(jump_prob, SimpleAdaptiveTauLeaping())
plot(sol)