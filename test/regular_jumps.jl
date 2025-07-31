using JumpProcesses, DiffEqBase
using Test, LinearAlgebra
using StableRNGs
rng = StableRNG(12345)

function regular_rate(out, u, p, t)
    out[1] = (0.1 / 1000.0) * u[1] * u[2]
    out[2] = 0.01u[2]
end

function regular_c(dc, u, p, t, mark)
    dc[1, 1] = -1
    dc[2, 1] = 1
    dc[2, 2] = -1
    dc[3, 2] = 1
end

dc = zeros(3, 2)

rj = RegularJump(regular_rate, regular_c, dc; constant_c = true)
jumps = JumpSet(rj)

prob = DiscreteProblem([999.0, 1.0, 0.0], (0.0, 250.0))
jump_prob = JumpProblem(prob, Direct(), rj; rng = rng)
sol = solve(jump_prob, SimpleTauLeaping(); dt = 1.0)

const _dc = zeros(3, 2)
dc[1, 1] = -1
dc[2, 1] = 1
dc[2, 2] = -1
dc[3, 2] = 1

function regular_c(du, u, p, t, counts, mark)
    mul!(du, dc, counts)
end

rj = RegularJump(regular_rate, regular_c, 2)
jumps = JumpSet(rj)
prob = DiscreteProblem([999, 1, 0], (0.0, 250.0))
jump_prob = JumpProblem(prob, Direct(), rj; rng = rng)
sol = solve(jump_prob, SimpleTauLeaping(); dt = 1.0)

# Decaying Dimerization Model
# Parameters
c1 = 1.0      # S1 -> 0
c2 = 10.0     # S1 + S1 <- S2
c3 = 1000.0   # S1 + S1 -> S2
c4 = 0.1      # S2 -> S3
p_dim = (c1, c2, c3, c4)

regular_rate_dim = (out, u, p, t) -> begin
    out[1] = p[1] * u[1]          # S1 -> 0
    out[2] = p[2] * u[2]          # S1 + S1 <- S2
    out[3] = p[3] * u[1] * (u[1] - 1) / 2  # S1 + S1 -> S2
    out[4] = p[4] * u[2]          # S2 -> S3
end

regular_c_dim = (du, u, p, t, counts, mark) -> begin
    du .= 0.0
    du[1] = -counts[1] - 2 * counts[3] + 2 * counts[2]  # S1: -decay - 2*forward + 2*backward
    du[2] = counts[3] - counts[2] - counts[4]           # S2: +forward - backward - decay
    du[3] = counts[4]                                   # S3: +decay
end

u0_dim = [10000.0, 0.0, 0.0]  # S1, S2, S3
tspan_dim = (0.0, 4.0)

prob_disc_dim = DiscreteProblem(u0_dim, tspan_dim, p_dim)
rj_dim = RegularJump(regular_rate_dim, regular_c_dim, 4)
jump_prob_dim = JumpProblem(prob_disc_dim, Direct(), rj_dim; rng=rng)

sol = solve(jump_prob_dim, ImplicitTauLeaping())