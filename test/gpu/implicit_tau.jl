using JumpProcesses, DiffEqBase
using Test, LinearAlgebra, Statistics
using KernelAbstractions, Adapt, CUDA
using StableRNGs


rng = StableRNG(12345)
Nsims = 1


# Decaying Dimerization Model
c1 = 1.0      # S1 -> 0
c2 = 10.0     # S1 + S1 <- S2
c3 = 1000.0   # S1 + S1 -> S2
c4 = 0.1      # S2 -> S3
p = (c1, c2, c3, c4)

# Propensity functions
regular_rate = (out, u, p, t) -> begin
    @assert typeof(p) == NTuple{4, Float64} "p must be a tuple of 4 Float64 values"
    out[1] = p[1] * u[1]          # S1 -> 0
    out[2] = p[2] * u[2]          # S1 + S1 <- S2
    out[3] = p[3] * u[1] * (u[1] - 1) / 2  # S1 + S1 -> S2
    out[4] = p[4] * u[2]          # S2 -> S3
end

# State change function
regular_c = (du, u, p, t, counts, mark) -> begin
    @assert typeof(p) == NTuple{4, Float64} "p must be a tuple of 4 Float64 values"
    du .= 0.0
    du[1] = -counts[1] - 2 * counts[3] + 2 * counts[2]  # S1: -decay - 2*forward + 2*backward
    du[2] = counts[3] - counts[2] - counts[4]           # S2: +forward - backward - decay
    du[3] = counts[4]                                   # S3: +decay
end

# Initial condition
u0 = [10000.0, 0.0, 0.0]  # S1, S2, S3
tspan = (0.0, 4.0)

# Create JumpProblem
prob_disc = DiscreteProblem(u0, tspan, p)
rj = RegularJump(regular_rate, regular_c, 4)
jump_prob = JumpProblem(prob_disc, Direct(), rj; rng=rng)

# Solve using ImplicitTauLeaping
sol = solve(EnsembleProblem(jump_prob), ImplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims)
