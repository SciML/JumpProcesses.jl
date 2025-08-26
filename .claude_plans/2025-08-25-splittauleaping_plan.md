# SimpleSplitTauLeaping Implementation Plan
Date: 2025-08-25

## Overview
Implementation of a new tau-leaping integrator, `SimpleSplitTauLeaping`, that uses first-order operator splitting to evaluate one jump at a time. This initial implementation focuses on MassActionJumps only, following the type-stability practices used in Direct() and other constant rate aggregators.

## 1. Algorithm Definition
Add to `src/simple_regular_solve.jl`:
```julia
struct SimpleSplitTauLeaping <: DiffEqBase.DEAlgorithm end
```

## 2. Validation Function
```julia
function validate_massjump_splitting_inputs(jump_prob::JumpProblem, alg)
    if !(jump_prob.aggregator isa PureLeaping)
        @warn "When using $alg, please pass PureLeaping() as the aggregator..."
    end
    # Only MassActionJumps allowed
    isempty(jump_prob.jump_callback.continuous_callbacks) &&
    isempty(jump_prob.jump_callback.discrete_callbacks) &&
    isempty(jump_prob.constant_jumps) &&
    isempty(jump_prob.variable_jumps) &&
    jump_prob.regular_jump === nothing &&
    get_num_majumps(jump_prob.massaction_jump) > 0
end
```

## 3. Core Implementation (Allocation-Free)
```julia
function DiffEqBase.solve(jump_prob::JumpProblem, alg::SimpleSplitTauLeaping;
        seed = nothing,
        dt = error("dt is required for SimpleSplitTauLeaping."))
    
    validate_massjump_splitting_inputs(jump_prob, alg) ||
        error("SimpleSplitTauLeaping currently only supports MassActionJumps with PureLeaping")
    
    prob = jump_prob.prob
    rng = DEFAULT_RNG
    (seed !== nothing) && seed!(rng, seed)
    
    # Extract MassActionJumps
    ma_jumps = jump_prob.massaction_jump
    num_jumps = get_num_majumps(ma_jumps)
    
    # Pre-allocate
    u0 = copy(prob.u0)
    u_work = similar(u0)  # Working state vector
    
    tspan = prob.tspan
    p = prob.p
    n = Int((tspan[2] - tspan[1]) / dt) + 1
    u = Vector{typeof(u0)}(undef, n)
    u[1] = u0
    t = tspan[1]:dt:tspan[2]
    
    # Main loop - operator splitting with individual jump execution
    for i in 2:n
        copy!(u_work, u[i-1])
        
        # Split tau-leaping: evaluate and execute each jump separately
        @inbounds for j in 1:num_jumps
            rate = evalrxrate(u_work, j, ma_jumps)
            num_firings = pois_rand(rng, rate * dt)
            
            # Execute this jump num_firings times
            for _ in 1:num_firings
                executerx!(u_work, j, ma_jumps)
            end
        end
        
        u[i] = copy(u_work)
    end
    
    sol = DiffEqBase.build_solution(prob, alg, t, u,
        calculate_error = false,
        interp = DiffEqBase.ConstantInterpolation(t, u))
end
```

## 4. Test Implementation
Create comprehensive tests with 5% accuracy target:

```julia
@testset "SimpleSplitTauLeaping MassActionJump Tests" begin
    using Statistics, JumpProcesses, DiffEqBase, Random
    
    # Test 1: Birth-death process via MassActionJumps
    @testset "Birth-Death MassActionJump" begin
        # ∅ → X (birth), X → ∅ (death)
        reactant_stoich = [Vector{Pair{Int,Int}}(),  # ∅ → X
                          [1 => 1]]                    # X → ∅
        net_stoich = [[1 => 1],                       # ∅ → X adds one
                      [1 => -1]]                       # X → ∅ removes one
        rates = [1.0, 0.1]  # birth rate, death rate
        
        ma_jumps = MassActionJump(rates, reactant_stoich, net_stoich)
        
        u0 = [10]
        tspan = (0.0, 100.0)
        dprob = DiscreteProblem(u0, tspan)
        
        # Run ensembles for statistics
        n_traj = 10000
        
        # Direct method reference
        jprob_direct = JumpProblem(dprob, Direct(), ma_jumps)
        ensembleprob_direct = EnsembleProblem(jprob_direct)
        sol_direct = solve(ensembleprob_direct, SSAStepper(), 
                          EnsembleThreads(), trajectories=n_traj)
        
        # SimpleSplitTauLeaping with small dt
        jprob_split = JumpProblem(dprob, PureLeaping(), ma_jumps)
        
        function run_split_ensemble(prob, n_traj, dt, seed)
            sols = Vector{Any}(undef, n_traj)
            for i in 1:n_traj
                sols[i] = solve(prob, SimpleSplitTauLeaping(), 
                               dt=dt, seed=seed+i)
            end
            return sols
        end
        
        sol_split = run_split_ensemble(jprob_split, n_traj, 0.001, 12345)
        
        # Extract final values
        direct_final = [sol.u[end][1] for sol in sol_direct]
        split_final = [sol.u[end][1] for sol in sol_split]
        
        # Test mean and variance (5% relative accuracy)
        @test mean(split_final) ≈ mean(direct_final) rtol=0.05
        @test var(split_final) ≈ var(direct_final) rtol=0.05
    end
    
    # Test 2: Simple reaction A + B → C
    @testset "A + B → C Reaction" begin
        reactant_stoich = [[1 => 1, 2 => 1]]  # A + B
        net_stoich = [[1 => -1, 2 => -1, 3 => 1]]  # -A -B +C
        rates = [0.001]
        
        ma_jumps = MassActionJump(rates, reactant_stoich, net_stoich)
        
        u0 = [100, 100, 0]
        tspan = (0.0, 10.0)
        dprob = DiscreteProblem(u0, tspan)
        
        n_traj = 5000
        
        # Direct reference
        jprob_direct = JumpProblem(dprob, Direct(), ma_jumps)
        ensembleprob_direct = EnsembleProblem(jprob_direct)
        sol_direct = solve(ensembleprob_direct, SSAStepper(),
                          EnsembleThreads(), trajectories=n_traj)
        
        # SimpleSplitTauLeaping
        jprob_split = JumpProblem(dprob, PureLeaping(), ma_jumps)
        sol_split = run_split_ensemble(jprob_split, n_traj, 0.0001, 54321)
        
        # Compare means of all species at final time
        for species in 1:3
            direct_vals = [sol.u[end][species] for sol in sol_direct]
            split_vals = [sol.u[end][species] for sol in sol_split]
            
            @test mean(split_vals) ≈ mean(direct_vals) rtol=0.05
            @test var(split_vals) ≈ var(direct_vals) rtol=0.05
        end
        
        # Check conservation
        for sol in sol_split
            @test sum(sol.u[end]) == sum(u0)
        end
    end
    
    # Test 3: Lotka-Volterra predator-prey
    @testset "Lotka-Volterra System" begin
        # X → 2X (prey birth)
        # X + Y → 2Y (predation) 
        # Y → ∅ (predator death)
        reactant_stoich = [[1 => 1],           # X
                          [1 => 1, 2 => 1],     # X + Y
                          [2 => 1]]             # Y
        net_stoich = [[1 => 1],                # X births
                      [1 => -1, 2 => 1],        # X dies, Y births
                      [2 => -1]]                # Y dies
        rates = [1.0, 0.001, 1.0]
        
        ma_jumps = MassActionJump(rates, reactant_stoich, net_stoich)
        
        u0 = [100, 100]  # Initial prey and predator
        tspan = (0.0, 20.0)
        dprob = DiscreteProblem(u0, tspan)
        
        n_traj = 5000
        
        # Direct
        jprob_direct = JumpProblem(dprob, Direct(), ma_jumps)
        ensembleprob_direct = EnsembleProblem(jprob_direct)
        sol_direct = solve(ensembleprob_direct, SSAStepper(),
                          EnsembleThreads(), trajectories=n_traj)
        
        # SimpleSplitTauLeaping with very small dt for accuracy
        jprob_split = JumpProblem(dprob, PureLeaping(), ma_jumps)
        sol_split = run_split_ensemble(jprob_split, n_traj, 0.0001, 99999)
        
        # Sample at multiple time points
        test_times = [5.0, 10.0, 15.0, 20.0]
        for test_t in test_times
            # Find closest time index
            t_idx_direct = findfirst(t -> t >= test_t, sol_direct[1].t)
            t_idx_split = findfirst(t -> t >= test_t, sol_split[1].t)
            
            for species in 1:2
                direct_vals = [sol.u[t_idx_direct][species] for sol in sol_direct]
                split_vals = [sol.u[t_idx_split][species] for sol in sol_split]
                
                @test mean(split_vals) ≈ mean(direct_vals) rtol=0.05
            end
        end
    end
end
```

## 5. Key Simplifications

- **MassActionJumps only**: No need for FunctionWrappers or type dispatch
- **Direct rate evaluation**: Use existing `evalrxrate` and `executerx!`
- **Minimal allocations**: Only `u_work` vector for in-place updates
- **Simple validation**: Check only for MassActionJumps presence

## 6. Performance Notes

- Type stable since only one jump type
- Cache-friendly sequential access
- Minimal branching in inner loop
- In-place operations throughout

## 7. Implementation Strategy

1. Add the struct and solve method to `src/simple_regular_solve.jl`
2. Add export in `src/JumpProcesses.jl`
3. Create test file or add to existing test suite
4. Verify 5% accuracy requirement is met across different models
5. Document the method's operator splitting approach

## 8. Future Extensions

Once the basic MassActionJump implementation is working:
- Add support for ConstantRateJumps using FunctionWrappers
- Add support for VariableRateJumps
- Consider adaptive time-stepping
- Optimize for specific reaction network structures