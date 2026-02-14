using JumpProcesses, DiffEqBase
using Test, LinearAlgebra, Statistics
using StableRNGs
rng = StableRNG(12345)

Nsims = 100

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
    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

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
    sol_adaptive = solve(EnsembleProblem(jump_prob_maj), SimpleExplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

    # Solve with SimpleImplicitTauLeaping (Newton)
    sol_implicit_newton = solve(EnsembleProblem(jump_prob_maj), SimpleImplicitTauLeaping(solver=NewtonImplicitSolver()), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

    # Solve with SimpleImplicitTauLeaping (Trapezoidal)
    sol_implicit_trap = solve(EnsembleProblem(jump_prob_maj), SimpleImplicitTauLeaping(solver=TrapezoidalImplicitSolver()), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

    # Simple test: Check that all solvers completed successfully and have reasonable output
    @test length(sol_direct) == Nsims
    @test length(sol_simple) == Nsims
    @test length(sol_adaptive) == Nsims
    @test length(sol_implicit_newton) == Nsims
    @test length(sol_implicit_trap) == Nsims
    
    # Check that final times match expected tspan
    @test sol_direct[1].t[end] ≈ 250.0 atol=1.0
    @test sol_simple[1].t[end] ≈ 250.0 atol=1.0
    @test sol_adaptive[1].t[end] ≈ 250.0 atol=1.0
    @test sol_implicit_newton[1].t[end] ≈ 250.0 atol=1.0
    @test sol_implicit_trap[1].t[end] ≈ 250.0 atol=1.0
    
    # Sample at key time points (0, 50, 100, 150, 200, 250)
    t_sample = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0]
    
    # Compute mean I at sample times for each method
    mean_I_direct = [mean(sol_direct[i](t)[2] for i in 1:Nsims) for t in t_sample]
    mean_I_simple = [mean(sol_simple[i](t)[2] for i in 1:Nsims) for t in t_sample]
    mean_I_explicit = [mean(sol_adaptive[i](t)[2] for i in 1:Nsims) for t in t_sample]
    mean_I_implicit_newton = [mean(sol_implicit_newton[i](t)[2] for i in 1:Nsims) for t in t_sample]
    mean_I_implicit_trap = [mean(sol_implicit_trap[i](t)[2] for i in 1:Nsims) for t in t_sample]
    
    # Check that mean infected values are in reasonable range (0 to population size)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_direct)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_simple)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_explicit)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_implicit_newton)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_implicit_trap)
    
    # Check that all methods produce similar dynamics (loose tolerance)
    @test isapprox(mean_I_direct[3], mean_I_simple[3], rtol=0.05)  # Compare at t=100
    @test isapprox(mean_I_direct[3], mean_I_explicit[3], rtol=0.05)
    @test isapprox(mean_I_direct[3], mean_I_implicit_newton[3], rtol=0.05)
    @test isapprox(mean_I_direct[3], mean_I_implicit_trap[3], rtol=0.05)
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
    sol_direct = solve(EnsembleProblem(jump_prob), SSAStepper(), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

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
    sol_adaptive = solve(EnsembleProblem(jump_prob_maj), SimpleExplicitTauLeaping(), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

    # Solve with SimpleImplicitTauLeaping (Newton)
    sol_implicit_newton = solve(EnsembleProblem(jump_prob_maj), SimpleImplicitTauLeaping(solver=NewtonImplicitSolver()), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

    # Solve with SimpleImplicitTauLeaping (Trapezoidal)
    sol_implicit_trap = solve(EnsembleProblem(jump_prob_maj), SimpleImplicitTauLeaping(solver=TrapezoidalImplicitSolver()), EnsembleSerial(); trajectories=Nsims, saveat=5.0)

    # Simple test: Check that all solvers completed successfully and have reasonable output
    @test length(sol_direct) == Nsims
    @test length(sol_simple) == Nsims
    @test length(sol_adaptive) == Nsims
    @test length(sol_implicit_newton) == Nsims
    @test length(sol_implicit_trap) == Nsims
    
    # Check that final times match expected tspan
    @test sol_direct[1].t[end] ≈ 250.0 atol=1.0
    @test sol_simple[1].t[end] ≈ 250.0 atol=1.0
    @test sol_adaptive[1].t[end] ≈ 250.0 atol=1.0
    @test sol_implicit_newton[1].t[end] ≈ 250.0 atol=1.0
    @test sol_implicit_trap[1].t[end] ≈ 250.0 atol=1.0
    
    # Sample at key time points (0, 50, 100, 150, 200, 250)
    t_sample = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0]
    
    # Compute mean I at sample times for each method (I is index 3 in SEIR)
    mean_I_direct = [mean(sol_direct[i](t)[3] for i in 1:Nsims) for t in t_sample]
    mean_I_simple = [mean(sol_simple[i](t)[3] for i in 1:Nsims) for t in t_sample]
    mean_I_explicit = [mean(sol_adaptive[i](t)[3] for i in 1:Nsims) for t in t_sample]
    mean_I_implicit_newton = [mean(sol_implicit_newton[i](t)[3] for i in 1:Nsims) for t in t_sample]
    mean_I_implicit_trap = [mean(sol_implicit_trap[i](t)[3] for i in 1:Nsims) for t in t_sample]
    
    # Check that mean infected values are in reasonable range (0 to population size)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_direct)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_simple)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_explicit)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_implicit_newton)
    @test all(0 ≤ m ≤ 1000 for m in mean_I_implicit_trap)
    
    # Check that all methods produce similar dynamics (loose tolerance)
    @test isapprox(mean_I_direct[3], mean_I_simple[3], rtol=0.05)  # Compare at t=100
    @test isapprox(mean_I_direct[3], mean_I_explicit[3], rtol=0.05)
    @test isapprox(mean_I_direct[3], mean_I_implicit_newton[3], rtol=0.05)
    @test isapprox(mean_I_direct[3], mean_I_implicit_trap[3], rtol=0.05)
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

    sol = solve(jump_prob, SimpleExplicitTauLeaping(); dtmin = 0.1, saveat=1.0)

    # Check that solution completes and covers tspan
    @test sol.t[end] ≈ 250.0 atol=1e-6
    # Check that state remains zero
    @test all(u == [0, 0, 0] for u in sol.u)
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

# Test implicit solvers on stiff system
@testset "Stiff System with Implicit Solvers" begin
    # Example system from Cao et al. (2007)
    # Reactions: S1 -> S2, S2 -> S1, S2 -> S3
    # Rate constants
    c = (1000.0, 1000.0, 1.0)
    
    # Define MassActionJump
    # Reaction 1: S1 -> S2
    reactant_stoich1 = [Pair(1, 1)]  # S1 consumed
    net_stoich1 = [Pair(1, -1), Pair(2, 1)]  # S1 -1, S2 +1
    # Reaction 2: S2 -> S1
    reactant_stoich2 = [Pair(2, 1)]  # S2 consumed
    net_stoich2 = [Pair(1, 1), Pair(2, -1)]  # S1 +1, S2 -1
    # Reaction 3: S2 -> S3
    reactant_stoich3 = [Pair(2, 1)]  # S2 consumed
    net_stoich3 = [Pair(2, -1), Pair(3, 1)]  # S2 -1, S3 +1
    
    maj = MassActionJump([c[1], c[2], c[3]], [reactant_stoich1, reactant_stoich2, reactant_stoich3], 
                          [net_stoich1, net_stoich2, net_stoich3])
    
    u0 = [100, 0, 0]  # Initial: S1=100, S2=0, S3=0
    tspan = (0.0, 1.0)
    prob = DiscreteProblem(u0, tspan)
    jump_prob = JumpProblem(prob, PureLeaping(), maj; rng=rng)
    
    # Solve with SimpleExplicitTauLeaping
    sol_explicit = solve(jump_prob, SimpleExplicitTauLeaping(); dtmin=1e-6, saveat=0.1)
    
    # Solve with SimpleImplicitTauLeaping (Newton) - should handle stiffness better
    sol_implicit_newton = solve(jump_prob, SimpleImplicitTauLeaping(solver=NewtonImplicitSolver()); dtmin=1e-6, saveat=0.1)
    
    # Solve with SimpleImplicitTauLeaping (Trapezoidal)
    sol_implicit_trap = solve(jump_prob, SimpleImplicitTauLeaping(solver=TrapezoidalImplicitSolver()); dtmin=1e-6, saveat=0.1)
    
    # Check that all solvers completed successfully
    @test sol_explicit.t[end] ≈ 1.0 atol=1e-3
    @test sol_implicit_newton.t[end] ≈ 1.0 atol=1e-3
    @test sol_implicit_trap.t[end] ≈ 1.0 atol=1e-3
    
    # Check conservation: S1 + S2 + S3 should equal initial total
    @test all(sum(u) ≈ 100 for u in sol_explicit.u)
    @test all(sum(u) ≈ 100 for u in sol_implicit_newton.u)
    @test all(sum(u) ≈ 100 for u in sol_implicit_trap.u)
    
    # Check that solutions are non-negative
    @test all(all(x >= 0 for x in u) for u in sol_explicit.u)
    @test all(all(x >= 0 for x in u) for u in sol_implicit_newton.u)
    @test all(all(x >= 0 for x in u) for u in sol_implicit_trap.u)
    
    # For stiff system with fast equilibration between S1 and S2, 
    # S3 should increase monotonically
    @test sol_explicit.u[end][3] >= sol_explicit.u[1][3]
    @test sol_implicit_newton.u[end][3] >= sol_implicit_newton.u[1][3]
    @test sol_implicit_trap.u[end][3] >= sol_implicit_trap.u[1][3]
end
