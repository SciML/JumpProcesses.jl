using JumpProcesses, DiffEqBase
using Test, LinearAlgebra, Statistics
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
