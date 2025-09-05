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
    rate3(u, p, t) = p[3]                # influx_rate (S influx)
    affect1!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    affect2!(integrator) = (integrator.u[2] -= 1; integrator.u[3] += 1; nothing)
    affect3!(integrator) = (integrator.u[1] += 1; nothing)
    jumps = (ConstantRateJump(rate1, affect1!), ConstantRateJump(rate2, affect2!), ConstantRateJump(rate3, affect3!))

    u0 = [999.0, 10.0, 0.0]  # S, I, R
    tspan = (0.0, 250.0)
    prob_disc = DiscreteProblem(u0, tspan, p)
    jump_prob = JumpProblem(prob_disc, Direct(), jumps...; rng=rng)

    # Solve with SSAStepper
    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims, saveat=1.0)

    # RegularJump formulation for SimpleTauLeaping
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
    jump_prob_tau = JumpProblem(prob_disc, PureLeaping(), rj; rng=rng)

    # Solve with SimpleTauLeaping
    sol_simple = solve(EnsembleProblem(jump_prob_tau), SimpleTauLeaping(), EnsembleSerial(); trajectories=Nsims, dt=0.1)

    # MassActionJump formulation for SimpleExplicitTauLeaping
    reactant_stoich = [[1=>1, 2=>1], [2=>1], Pair{Int,Int}[]]
    net_stoich = [[1=>-1, 2=>1], [2=>-1, 3=>1], [1=>1]]
    param_idxs = [1, 2, 3]
    maj = MassActionJump(reactant_stoich, net_stoich; param_idxs=param_idxs)
    jump_prob_maj = JumpProblem(prob_disc, PureLeaping(), maj; rng=rng)

    # Solve with SimpleExplicitTauLeaping
    sol_adaptive = solve(EnsembleProblem(jump_prob_maj), SimpleExplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat=1.0)

    # Compute mean infected (I) trajectories
    t_points = 0:1.0:250.0
    mean_direct_I = [mean(sol_direct[i](t)[2] for i in 1:Nsims) for t in t_points]
    mean_simple_I = [mean(sol_simple[i](t)[2] for i in 1:Nsims) for t in t_points]
    mean_adaptive_I = [mean(sol_adaptive[i](t)[2] for i in 1:Nsims) for t in t_points]

    # Test mean infected trajectories
    for i in 1:10:251
        @test isapprox(mean_direct_I[i], mean_simple_I[i], rtol=0.05)
        @test isapprox(mean_direct_I[i], mean_adaptive_I[i], rtol=0.05)
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
    jump_prob = JumpProblem(prob_disc, Direct(), jumps...; rng=rng)

    # Solve with SSAStepper
    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims, saveat=1.0)

    # RegularJump formulation for SimpleTauLeaping
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
    jump_prob_tau = JumpProblem(prob_disc, PureLeaping(), rj; rng=rng)

    # Solve with SimpleTauLeaping
    sol_simple = solve(EnsembleProblem(jump_prob_tau), SimpleTauLeaping(), EnsembleSerial(); trajectories=Nsims, dt=0.1)

    # MassActionJump formulation for SimpleExplicitTauLeaping
    reactant_stoich = [[1=>1, 3=>1], [2=>1], [3=>1]]
    net_stoich = [[1=>-1, 2=>1], [2=>-1, 3=>1], [3=>-1, 4=>1]]
    param_idxs = [1, 2, 3]
    maj = MassActionJump(reactant_stoich, net_stoich; param_idxs=param_idxs)
    jump_prob_maj = JumpProblem(prob_disc, PureLeaping(), maj; rng=rng)

    # Solve with SimpleExplicitTauLeaping
    sol_adaptive = solve(EnsembleProblem(jump_prob_maj), SimpleExplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat=1.0)

    # Compute mean infected (I) trajectories
    t_points = 0:1.0:250.0
    mean_direct_I = [mean(sol_direct[i](t)[3] for i in 1:Nsims) for t in t_points]
    mean_simple_I = [mean(sol_simple[i](t)[3] for i in 1:Nsims) for t in t_points]
    mean_adaptive_I = [mean(sol_adaptive[i](t)[3] for i in 1:Nsims) for t in t_points]

    # Test mean infected trajectories
    for i in 1:10:251
        @test isapprox(mean_direct_I[i], mean_simple_I[i], rtol=0.05)
        @test isapprox(mean_direct_I[i], mean_adaptive_I[i], rtol=0.05)
    end
end

# Test zero-rate case for SimpleExplicitTauLeaping
@testset "Zero Rates Test for SimpleExplicitTauLeaping" begin
    # SIR model: S + I -> 2I, I -> R
    reactant_stoch = [[1=>1, 2=>1], [2=>1], Pair{Int,Int}[]]
    net_stoch = [[1=>-1, 2=>1], [2=>-1, 3=>1], []]
    rates = [0.1/1000, 0.05, 0.0]  # beta/N, gamma, dummy rate for empty reaction
    maj = MassActionJump(rates, reactant_stoch, net_stoch)
    u0 = [0, 0, 0]  # All populations zero
    tspan = (0.0, 250.0)
    prob = DiscreteProblem(u0, tspan)
    jump_prob = JumpProblem(prob, PureLeaping(), maj)

    sol = solve(EnsembleProblem(jump_prob), SimpleExplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, dtmin = 0.1, saveat=1.0)
    
    for i in 1:Nsims
        # Check that solution completes and covers tspan
        @test sol[i].t[end] ≈ 250.0 atol=1e-6
        # Check that state remains zero
        @test all(u == [0, 0, 0] for u in sol[i].u)
    end
end

# Test PureLeaping aggregator functionality
@testset "PureLeaping Aggregator Tests" begin
    # Test with MassActionJump
    u0 = [10, 5, 0]
    tspan = (0.0, 10.0)
    p = [0.1, 0.2]
    prob = DiscreteProblem(u0, tspan, p)
    
    # Create MassActionJump
    reactant_stoich = [[1 => 1], [1 => 2]]
    net_stoich = [[1 => -1, 2 => 1], [1 => -2, 3 => 1]]
    rates = [0.1, 0.05]
    maj = MassActionJump(rates, reactant_stoich, net_stoich)
    
    # Test PureLeaping JumpProblem creation
    jp_pure = JumpProblem(prob, PureLeaping(), JumpSet(maj); rng)
    @test jp_pure.aggregator isa PureLeaping
    @test jp_pure.discrete_jump_aggregation === nothing
    @test jp_pure.massaction_jump !== nothing
    @test length(jp_pure.jump_callback.discrete_callbacks) == 0
    
    # Test with ConstantRateJump
    rate(u, p, t) = p[1] * u[1]
    affect!(integrator) = (integrator.u[1] -= 1; integrator.u[3] += 1)
    crj = ConstantRateJump(rate, affect!)
    
    jp_pure_crj = JumpProblem(prob, PureLeaping(), JumpSet(crj); rng)
    @test jp_pure_crj.aggregator isa PureLeaping
    @test jp_pure_crj.discrete_jump_aggregation === nothing
    @test length(jp_pure_crj.constant_jumps) == 1
    
    # Test with VariableRateJump
    vrate(u, p, t) = t * p[1] * u[1]
    vaffect!(integrator) = (integrator.u[1] -= 1; integrator.u[3] += 1)
    vrj = VariableRateJump(vrate, vaffect!)
    
    jp_pure_vrj = JumpProblem(prob, PureLeaping(), JumpSet(vrj); rng)
    @test jp_pure_vrj.aggregator isa PureLeaping
    @test jp_pure_vrj.discrete_jump_aggregation === nothing
    @test length(jp_pure_vrj.variable_jumps) == 1
    
    # Test with RegularJump
    function rj_rate(out, u, p, t)
        out[1] = p[1] * u[1]
    end
    
    rj_dc = zeros(3, 1)
    rj_dc[1, 1] = -1
    rj_dc[3, 1] = 1
    
    function rj_c(du, u, p, t, counts, mark)
        mul!(du, rj_dc, counts)
    end
    
    regj = RegularJump(rj_rate, rj_c, 1)
    
    jp_pure_regj = JumpProblem(prob, PureLeaping(), JumpSet(regj); rng)
    @test jp_pure_regj.aggregator isa PureLeaping
    @test jp_pure_regj.discrete_jump_aggregation === nothing
    @test jp_pure_regj.regular_jump !== nothing
    
    # Test mixed jump types
    mixed_jumps = JumpSet(; massaction_jumps = maj, constant_jumps = (crj,), 
        variable_jumps = (vrj,), regular_jumps = regj)
    jp_pure_mixed = JumpProblem(prob, PureLeaping(), mixed_jumps; rng)
    @test jp_pure_mixed.aggregator isa PureLeaping
    @test jp_pure_mixed.discrete_jump_aggregation === nothing
    @test jp_pure_mixed.massaction_jump !== nothing
    @test length(jp_pure_mixed.constant_jumps) == 1
    @test length(jp_pure_mixed.variable_jumps) == 1
    @test jp_pure_mixed.regular_jump !== nothing
    
    # Test spatial system error
    spatial_sys = CartesianGrid((2, 2))
    hopping_consts = [1.0]
    @test_throws ErrorException JumpProblem(prob, PureLeaping(), JumpSet(maj); rng,
                                          spatial_system = spatial_sys)
    @test_throws ErrorException JumpProblem(prob, PureLeaping(), JumpSet(maj); rng,
                                          hopping_constants = hopping_consts)
    
    # Test MassActionJump with parameter mapping
    maj_params = MassActionJump(reactant_stoich, net_stoich; param_idxs = [1, 2])
    jp_params = JumpProblem(prob, PureLeaping(), JumpSet(maj_params); rng)
    scaled_rates = [p[1], p[2]/2]
    @test jp_params.massaction_jump.scaled_rates == scaled_rates
end
