using JumpProcesses, DiffEqBase
using Test, LinearAlgebra, Statistics
using StableRNGs
rng = StableRNG(12345)

Nsims = 8000

@testset "SIR Model Correctness" begin
    β = 0.1 / 1000.0
    ν = 0.01
    influx_rate = 1.0
    p = (β, ν, influx_rate)

    rate1(u, p, t) = p[1] * u[1] * u[2]
    rate2(u, p, t) = p[2] * u[2]
    rate3(u, p, t) = p[3]
    affect1!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    affect2!(integrator) = (integrator.u[2] -= 1; integrator.u[3] += 1; nothing)
    affect3!(integrator) = (integrator.u[1] += 1; nothing)
    jumps = (ConstantRateJump(rate1, affect1!), ConstantRateJump(rate2, affect2!), ConstantRateJump(rate3, affect3!))

    u0 = [999, 10, 0]  # Integer initial conditions
    tspan = (0.0, 250.0)
    prob_disc = DiscreteProblem(u0, tspan, p)
    jump_prob = JumpProblem(prob_disc, Direct(), jumps...; rng=StableRNG(12345))

    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims)

    regular_rate = (out, u, p, t) -> begin
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
    rj = RegularJump(regular_rate, regular_c, 3)
    jump_prob_tau = JumpProblem(prob_disc, Direct(), rj; rng=StableRNG(12345))

    sol_implicit = solve(EnsembleProblem(jump_prob_tau), SimpleImplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat=1.0)

    t_points = 0:1.0:250.0
    mean_direct_S = [mean(sol_direct[i](t)[1] for i in 1:Nsims) for t in t_points]
    mean_implicit_S = [mean(sol_implicit[i](t)[1] for i in 1:Nsims) for t in t_points]

    max_error_implicit = maximum(abs.(mean_direct_S .- mean_implicit_S))
    @test max_error_implicit < 0.01 * mean(mean_direct_S)
end

@testset "SEIR Model Correctness" begin
    β = 0.3 / 1000.0
    σ = 0.2
    ν = 0.01
    p = (β, σ, ν)

    rate1(u, p, t) = p[1] * u[1] * u[3]
    rate2(u, p, t) = p[2] * u[2]
    rate3(u, p, t) = p[3] * u[3]
    affect1!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    affect2!(integrator) = (integrator.u[2] -= 1; integrator.u[3] += 1; nothing)
    affect3!(integrator) = (integrator.u[3] -= 1; integrator.u[4] += 1; nothing)
    jumps = (ConstantRateJump(rate1, affect1!), ConstantRateJump(rate2, affect2!), ConstantRateJump(rate3, affect3!))

    u0 = [999, 0, 10, 0]  # Integer initial conditions
    tspan = (0.0, 250.0)
    prob_disc = DiscreteProblem(u0, tspan, p)
    jump_prob = JumpProblem(prob_disc, Direct(), jumps...; rng=StableRNG(12345))

    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims)

    regular_rate = (out, u, p, t) -> begin
        out[1] = p[1] * u[1] * u[3]
        out[2] = p[2] * u[2]
        out[3] = p[3] * u[3]
    end
    regular_c = (dc, u, p, t, counts, mark) -> begin
        dc .= 0
        dc[1] = -counts[1]
        dc[2] = counts[1] - counts[2]
        dc[3] = counts[2] - counts[3]
        dc[4] = counts[3]
    end
    rj = RegularJump(regular_rate, regular_c, 3)
    jump_prob_tau = JumpProblem(prob_disc, Direct(), rj; rng=StableRNG(12345))

    sol_implicit = solve(EnsembleProblem(jump_prob_tau), SimpleImplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat=1.0)

    t_points = 0:1.0:250.0
    mean_direct_R = [mean(sol_direct[i](t)[4] for i in 1:Nsims) for t in t_points]
    mean_implicit_R = [mean(sol_implicit[i](t)[4] for i in 1:Nsims) for t in t_points]

    max_error_implicit = maximum(abs.(mean_direct_R .- mean_implicit_R))
    @test max_error_implicit < 0.01 * mean(mean_direct_R)
end
