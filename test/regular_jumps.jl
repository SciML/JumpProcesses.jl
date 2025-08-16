using JumpProcesses, DiffEqBase
using Test, LinearAlgebra, Statistics
using StableRNGs
rng = StableRNG(12345)

Nsims = 1000

# SIR model with influx
@testset "SIR Model Correctness" begin
    β = 0.1 / 1000.0
    ν = 0.01
    influx_rate = 1.0
    p = (β, ν, influx_rate)

    # ConstantRateJump formulation for SSAStepper
    rate1(u, p, t) = p[1] * u[1] * u[2]  # β*S*I (infection)
    rate2(u, p, t) = p[2] * u[2]         # ν*I (recovery)
    rate3(u, p, t) = p[3]                # influx_rate
    affect1!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    affect2!(integrator) = (integrator.u[2] -= 1; integrator.u[3] += 1; nothing)
    affect3!(integrator) = (integrator.u[1] += 1; nothing)
    jumps = (ConstantRateJump(rate1, affect1!), ConstantRateJump(rate2, affect2!), ConstantRateJump(rate3, affect3!))

    u0 = [999.0, 10.0, 0.0]  # S, I, R
    tspan = (0.0, 250.0)
    prob_disc = DiscreteProblem(u0, tspan, p)
    jump_prob = JumpProblem(prob_disc, Direct(), jumps...; rng = rng)

    # Solve with SSAStepper
    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims)

    # RegularJump formulation for TauLeaping methods
    regular_rate = (out, u, p, t) -> begin
        out[1] = p[1] * u[1] * u[2]
        out[2] = p[2] * u[2]
        out[3] = p[3]
    end
    regular_c = (dc, u, p, t, counts, mark) -> begin
        dc .= 0.0
        dc[1] = -counts[1] + counts[3]
        dc[2] = counts[1] - counts[2]
        dc[3] = counts[2]
    end
    rj = RegularJump(regular_rate, regular_c, 3)
    jump_prob_tau = JumpProblem(prob_disc, Direct(), rj; rng = rng)

    # Solve with SimpleTauLeaping (dt=0.1)
    sol_simple = solve(EnsembleProblem(jump_prob_tau), SimpleTauLeaping(), EnsembleSerial(); trajectories=Nsims, dt=0.1)
    
    # Solve with SimpleAdaptiveTauLeaping
    sol_adaptive = solve(EnsembleProblem(jump_prob_tau), SimpleAdaptiveTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat = 1.0)

    # Compute mean trajectories at t = 0, 1, ..., 250
    t_points = 0:1.0:250.0
    mean_direct_S = [mean(sol_direct[i](t)[2] for i in 1:Nsims) for t in t_points]
    mean_simple_S = [mean(sol_simple[i](t)[2] for i in 1:Nsims) for t in t_points]
    mean_adaptive_S = [mean(sol_adaptive[i](t)[2] for i in 1:Nsims) for t in t_points]

    for i in 1:251
        @test isapprox(mean_direct_S[i], mean_simple_S[i], rtol=0.10)
        @test isapprox(mean_direct_S[i], mean_adaptive_S[i], rtol=0.10)
    end
end

# SEIR model with exposed compartment
@testset "SEIR Model Correctness" begin
    β = 0.3 / 1000.0
    σ = 0.2
    ν = 0.01
    p = (β, σ, ν)

    # ConstantRateJump formulation for SSAStepper
    rate1(u, p, t) = p[1] * u[1] * u[3]  # β*S*I (infection)
    rate2(u, p, t) = p[2] * u[2]         # σ*E (progression)
    rate3(u, p, t) = p[3] * u[3]         # ν*I (recovery)
    affect1!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    affect2!(integrator) = (integrator.u[2] -= 1; integrator.u[3] += 1; nothing)
    affect3!(integrator) = (integrator.u[3] -= 1; integrator.u[4] += 1; nothing)
    jumps = (ConstantRateJump(rate1, affect1!), ConstantRateJump(rate2, affect2!), ConstantRateJump(rate3, affect3!))

    u0 = [999.0, 0.0, 10.0, 0.0]  # S, E, I, R
    tspan = (0.0, 250.0)
    prob_disc = DiscreteProblem(u0, tspan, p)
    jump_prob = JumpProblem(prob_disc, Direct(), jumps...; rng = rng)

    # Solve with SSAStepper
    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims)

    # RegularJump formulation for TauLeaping methods
    regular_rate = (out, u, p, t) -> begin
        out[1] = p[1] * u[1] * u[3]
        out[2] = p[2] * u[2]
        out[3] = p[3] * u[3]
    end
    regular_c = (dc, u, p, t, counts, mark) -> begin
        dc .= 0.0
        dc[1] = -counts[1]
        dc[2] = counts[1] - counts[2]
        dc[3] = counts[2] - counts[3]
        dc[4] = counts[3]
    end
    rj = RegularJump(regular_rate, regular_c, 3)
    jump_prob_tau = JumpProblem(prob_disc, Direct(), rj; rng = rng)

    # Solve with SimpleTauLeaping (dt=0.1)
    sol_simple = solve(EnsembleProblem(jump_prob_tau), SimpleTauLeaping(), EnsembleSerial(); trajectories=Nsims, dt=0.1)
    
    # Solve with SimpleAdaptiveTauLeaping
    sol_adaptive = solve(EnsembleProblem(jump_prob_tau), SimpleAdaptiveTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat = 1.0)

    # Compute mean trajectories at t = 0, 1, ..., 250
    t_points = 0:1.0:250.0
    mean_direct_S = [mean(sol_direct[i](t)[3] for i in 1:Nsims) for t in t_points]
    mean_simple_S = [mean(sol_simple[i](t)[3] for i in 1:Nsims) for t in t_points]
    mean_adaptive_S = [mean(sol_adaptive[i](t)[3] for i in 1:Nsims) for t in t_points]

    for i in 1:251
        @test isapprox(mean_direct_S[i], mean_simple_S[i], rtol=0.10)
        @test isapprox(mean_direct_S[i], mean_adaptive_S[i], rtol=0.10)
    end
end
