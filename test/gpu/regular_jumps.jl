using JumpProcesses, DiffEqBase
using Test, LinearAlgebra, Statistics
using KernelAbstractions, Adapt, CUDA
using StableRNGs
rng = StableRNG(12345)

Nsims = 100

# SIR model with influx
let
    β = 0.1 / 1000.0
    ν = 0.01
    influx_rate = 1.0
    p = (β, ν, influx_rate)

    regular_rate = (out, u, p, t) -> begin
        out[1] = p[1] * u[1] * u[2]  # β*S*I (infection)
        out[2] = p[2] * u[2]         # ν*I (recovery)
        out[3] = p[3]                # influx_rate
    end

    regular_c = (dc, u, p, t, counts, mark) -> begin
        dc .= 0.0
        dc[1] = -counts[1] + counts[3]  # S: -infection + influx
        dc[2] = counts[1] - counts[2]   # I: +infection - recovery
        dc[3] = counts[2]               # R: +recovery
    end

    u0 = [999.0, 10.0, 0.0]  # S, I, R
    tspan = (0.0, 250.0)

    prob_disc = DiscreteProblem(u0, tspan, p)
    rj = RegularJump(regular_rate, regular_c, 3)
    jump_prob = JumpProblem(prob_disc, Direct(), rj)

    sol = solve(EnsembleProblem(jump_prob), SimpleTauLeaping(),
        EnsembleGPUKernel(CUDABackend()); trajectories = Nsims, dt = 1.0)
    mean_kernel = mean(sol.u[i][1, end] for i in 1:Nsims)

    sol = solve(EnsembleProblem(jump_prob), SimpleTauLeaping(),
        EnsembleSerial(); trajectories = Nsims, dt = 1.0)
    mean_serial = mean(sol.u[i][1, end] for i in 1:Nsims)

    @test isapprox(mean_kernel, mean_serial, rtol = 0.05)
end

# SEIR model with exposed compartment
let
    β = 0.3 / 1000.0
    σ = 0.2
    ν = 0.01
    p = (β, σ, ν)

    regular_rate = (out, u, p, t) -> begin
        out[1] = p[1] * u[1] * u[3]  # β*S*I (infection)
        out[2] = p[2] * u[2]         # σ*E (progression)
        out[3] = p[3] * u[3]         # ν*I (recovery)
    end

    regular_c = (dc, u, p, t, counts, mark) -> begin
        dc .= 0.0
        dc[1] = -counts[1]           # S: -infection
        dc[2] = counts[1] - counts[2] # E: +infection - progression
        dc[3] = counts[2] - counts[3] # I: +progression - recovery
        dc[4] = counts[3]            # R: +recovery
    end

    # Initial state
    u0 = [999.0, 0.0, 10.0, 0.0]  # S, E, I, R
    tspan = (0.0, 250.0)

    # Create JumpProblem
    prob_disc = DiscreteProblem(u0, tspan, p)
    rj = RegularJump(regular_rate, regular_c, 3)
    jump_prob = JumpProblem(prob_disc, Direct(), rj; rng = StableRNG(12345))

    sol = solve(EnsembleProblem(jump_prob), SimpleTauLeaping(),
        EnsembleGPUKernel(CUDABackend()); trajectories = Nsims, dt = 1.0)
    mean_kernel = mean(sol.u[i][end, end] for i in 1:Nsims)

    sol = solve(EnsembleProblem(jump_prob), SimpleTauLeaping(),
        EnsembleSerial(); trajectories = Nsims, dt = 1.0)
    mean_serial = mean(sol.u[i][end, end] for i in 1:Nsims)

    @test isapprox(mean_kernel, mean_serial, rtol = 0.05)
end
