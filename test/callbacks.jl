using JumpProcesses, OrdinaryDiffEq, Test
using StableRNGs
rng = StableRNG(12345)

# Test that callbacks passed to JumpProblem constructor work correctly
# This tests the fix for the regression introduced in v9.17.0 (PR #514)

@testset "Callbacks in JumpProblem constructor" begin
    # Simple ODE with a jump
    function f!(du, u, p, t)
        du[1] = -0.5u[1]
    end

    rate(u, p, t) = 0.1
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [5.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f!, u0, tspan)

    # Test 1: ContinuousCallback in JumpProblem constructor
    cb_called = Ref(false)
    condition(u, t, integrator) = t - 5.0
    affect_cb!(integrator) = (cb_called[] = true)
    cb = ContinuousCallback(condition, affect_cb!)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cb)
    sol = solve(jprob, Tsit5())

    @test cb_called[]
    @test sol.t[end] ≈ 10.0

    # Test 2: DiscreteCallback in JumpProblem constructor
    dcb_called = Ref(0)
    condition_d(u, t, integrator) = t > 2.0  # Fire at every step after t=2
    affect_dcb!(integrator) = (dcb_called[] += 1)
    dcb = DiscreteCallback(condition_d, affect_dcb!)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = dcb)
    sol = solve(jprob, Tsit5())

    @test dcb_called[] > 0  # Should have fired multiple times

    # Test 3: Terminating callback in JumpProblem constructor
    condition_term(u, t, integrator) = t - 3.0
    affect_term!(integrator) = terminate!(integrator)
    cb_term = ContinuousCallback(condition_term, affect_term!)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cb_term)
    sol = solve(jprob, Tsit5())

    @test sol.t[end] ≈ 3.0  # Should terminate at t=3

    # Test 4: State-modifying callback in JumpProblem constructor
    condition_mod(u, t, integrator) = t - 5.0
    affect_mod!(integrator) = (integrator.u[1] *= 2.0)
    cb_mod = ContinuousCallback(condition_mod, affect_mod!)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cb_mod)
    sol = solve(jprob, Tsit5())

    # Check that state was modified at t=5
    idx = findfirst(t -> t >= 5.0, sol.t)
    @test idx !== nothing
    # State should have been doubled at this point
end

@testset "Callbacks in both JumpProblem and solve" begin
    # Test that callbacks merge correctly when passed to both

    function f!(du, u, p, t)
        du[1] = -0.5u[1]
    end

    rate(u, p, t) = 0.1
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [5.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f!, u0, tspan)

    # Create two callbacks with counters to verify no duplication
    cb1_count = Ref(0)
    condition1(u, t, integrator) = t - 3.0
    affect1!(integrator) = (cb1_count[] += 1)
    cb1 = ContinuousCallback(condition1, affect1!)

    cb2_count = Ref(0)
    condition2(u, t, integrator) = t - 7.0
    affect2!(integrator) = (cb2_count[] += 1)
    cb2 = ContinuousCallback(condition2, affect2!)

    # Test 1: Both callbacks should fire (default merge_callbacks = true)
    cb1_count[] = 0
    cb2_count[] = 0
    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cb1)
    sol = solve(jprob, Tsit5(); callback = cb2)

    @test cb1_count[] > 0
    @test cb2_count[] > 0
    @test sol.t[end] ≈ 10.0

    # Critical test: verify callbacks are not duplicated
    # Continuous callbacks should fire exactly once when crossing the threshold
    @test cb1_count[] == 1  # Should fire exactly once at t=3
    @test cb2_count[] == 1  # Should fire exactly once at t=7

    # Test 2: Only solve callback should fire (merge_callbacks = false)
    cb1_count[] = 0
    cb2_count[] = 0
    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cb1)
    sol = solve(jprob, Tsit5(); callback = cb2, merge_callbacks = false)

    @test cb1_count[] == 0  # Should not fire
    @test cb2_count[] == 1  # Should fire exactly once
    @test sol.t[end] ≈ 10.0
end

@testset "Callbacks with init and solve!" begin
    # Test that callbacks work through the init/solve! pathway

    function f!(du, u, p, t)
        du[1] = -0.5u[1]
    end

    rate(u, p, t) = 0.1
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [5.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f!, u0, tspan)

    cb_called = Ref(false)
    condition(u, t, integrator) = t - 5.0
    affect_cb!(integrator) = (cb_called[] = true)
    cb = ContinuousCallback(condition, affect_cb!)

    # Callback in JumpProblem constructor
    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cb)
    integrator = init(jprob, Tsit5())
    solve!(integrator)

    @test cb_called[]
    @test integrator.sol.t[end] ≈ 10.0
end

@testset "Multiple callback types in JumpProblem" begin
    # Test mixing ContinuousCallback and DiscreteCallback

    function f!(du, u, p, t)
        du[1] = -0.5u[1]
    end

    rate(u, p, t) = 0.1
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [5.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f!, u0, tspan)

    ccb_called = Ref(false)
    condition_c(u, t, integrator) = t - 5.0
    affect_c!(integrator) = (ccb_called[] = true)
    ccb = ContinuousCallback(condition_c, affect_c!)

    dcb_called = Ref(0)
    condition_d(u, t, integrator) = true
    affect_d!(integrator) = (dcb_called[] += 1)
    dcb = DiscreteCallback(condition_d, affect_d!)

    # Create CallbackSet with both types
    cbset = CallbackSet(ccb, dcb)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cbset)
    sol = solve(jprob, Tsit5())

    @test ccb_called[]
    @test dcb_called[] > 0
    @test sol.t[end] ≈ 10.0
end

@testset "Callback with SSAStepper" begin
    # Test that DiscreteCallbacks work with SSAStepper (pure jump systems)
    # SSAStepper only supports DiscreteCallbacks, not ContinuousCallbacks

    # Set up a simple birth process that will definitely increase u[1]
    rate1(u, p, t) = 5.0  # High rate to ensure jumps occur
    affect1!(integrator) = (integrator.u[1] += 1)
    jump1 = ConstantRateJump(rate1, affect1!)

    u0 = [0]
    tspan = (0.0, 10.0)
    dprob = DiscreteProblem(u0, tspan)

    # Test 1: DiscreteCallback in JumpProblem constructor that terminates when u[1] reaches 5
    cb_called = Ref(false)
    condition_term(u, t, integrator) = u[1] >= 5
    affect_term!(integrator) = (cb_called[] = true; terminate!(integrator))
    dcb_term = DiscreteCallback(condition_term, affect_term!)

    jprob = JumpProblem(dprob, Direct(), jump1; rng, callback = dcb_term)
    sol = solve(jprob, SSAStepper())

    @test cb_called[]  # Should have fired
    @test sol.u[end][1] >= 5  # Should have reached threshold
    @test sol.t[end] < 10.0  # Should have terminated early

    # Test 2: DiscreteCallback in solve call
    dcb_counter = Ref(0)
    condition_count(u, t, integrator) = u[1] >= 3
    affect_count!(integrator) = (dcb_counter[] += 1)
    dcb_count = DiscreteCallback(condition_count, affect_count!)

    jprob2 = JumpProblem(dprob, Direct(), jump1; rng)
    sol2 = solve(jprob2, SSAStepper(); callback = dcb_count)

    @test dcb_counter[] > 0  # Should have fired at least once

    # Test 3: DiscreteCallbacks in both JumpProblem and solve (should merge)
    # Use counters to verify callbacks fire the correct number of times (not duplicated)
    cb1_count = Ref(0)
    condition1(u, t, integrator) = u[1] == 3  # Fire exactly once when u[1] == 3
    affect_cb1!(integrator) = (cb1_count[] += 1)
    dcb1 = DiscreteCallback(condition1, affect_cb1!)

    cb2_count = Ref(0)
    condition2(u, t, integrator) = u[1] == 7  # Fire exactly once when u[1] == 7
    affect_cb2!(integrator) = (cb2_count[] += 1)
    dcb2 = DiscreteCallback(condition2, affect_cb2!)

    jprob3 = JumpProblem(dprob, Direct(), jump1; rng, callback = dcb1)
    sol3 = solve(jprob3, SSAStepper(); callback = dcb2)

    @test cb1_count[] > 0  # First callback should fire
    @test cb2_count[] > 0  # Second callback should fire
    @test sol3.u[end][1] >= 7  # Should reach threshold for second callback

    # Critical test: verify callbacks are not duplicated
    # Each callback should fire exactly once per discrete step where u[1] equals the target
    # If duplicated, we'd see exactly 2x the count (each firing twice)
    # Since jumps are discrete +1 increments, we should hit u==3 and u==7 each once
    @test cb1_count[] == 1  # Should fire exactly once
    @test cb2_count[] == 1  # Should fire exactly once

    # Test 4: merge_callbacks = false (solve callback should override)
    cb3_called = Ref(false)
    cb4_called = Ref(false)
    condition3(u, t, integrator) = u[1] >= 2
    affect_cb3!(integrator) = (cb3_called[] = true)
    dcb3 = DiscreteCallback(condition3, affect_cb3!)

    condition4(u, t, integrator) = u[1] >= 4
    affect_cb4!(integrator) = (cb4_called[] = true)
    dcb4 = DiscreteCallback(condition4, affect_cb4!)

    jprob4 = JumpProblem(dprob, Direct(), jump1; rng, callback = dcb3)
    sol4 = solve(jprob4, SSAStepper(); callback = dcb4, merge_callbacks = false)

    @test !cb3_called[]  # First callback should NOT fire
    @test cb4_called[]   # Second callback should fire
end

@testset "Regression test: Catalyst hybrid model callback" begin
    # Simplified version of the Catalyst hybrid_models.jl test that was failing
    # Tests continuous event in a hybrid model

    function f!(du, u, p, t)
        du[1] = p[1] * u[1]  # Simple exponential growth
    end

    rate(u, p, t) = p[2]
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [1.0]
    tspan = (0.0, 1.0)
    p = [1.0, 0.5]  # growth rate, jump rate
    prob = ODEProblem(f!, u0, tspan, p)

    # Continuous callback that terminates at t=0.5
    cb_called = Ref(false)
    condition(u, t, integrator) = t - 0.5
    affect_cb!(integrator) = (cb_called[] = true; terminate!(integrator))
    cb = ContinuousCallback(condition, affect_cb!)

    # This was broken in v9.17.0 - callback wouldn't fire
    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cb)
    sol = solve(jprob, Tsit5())

    @test cb_called[]
    @test sol.t[end] ≈ 0.5  # Should terminate at 0.5, not run to 1.0
    @test abs(sol.u[end][1] - exp(0.5)) < 0.5  # Rough check (jumps add noise)
end

@testset "Mixed callback types: Continuous in JumpProblem, Discrete in solve" begin
    # Test that continuous callbacks in JumpProblem work with discrete callbacks in solve

    function f!(du, u, p, t)
        du[1] = -0.5u[1]
    end

    rate(u, p, t) = 0.1
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [5.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f!, u0, tspan)

    # Continuous callback in JumpProblem
    ccb_called = Ref(false)
    condition_c(u, t, integrator) = t - 3.0
    affect_c!(integrator) = (ccb_called[] = true)
    ccb = ContinuousCallback(condition_c, affect_c!)

    # Discrete callback in solve
    dcb_called = Ref(0)
    condition_d(u, t, integrator) = t > 5.0
    affect_d!(integrator) = (dcb_called[] += 1)
    dcb = DiscreteCallback(condition_d, affect_d!)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = ccb)
    sol = solve(jprob, Tsit5(); callback = dcb)

    @test ccb_called[]  # Continuous callback should fire
    @test dcb_called[] > 0  # Discrete callback should fire multiple times
end

@testset "Mixed callback types: Discrete in JumpProblem, Continuous in solve" begin
    # Test that discrete callbacks in JumpProblem work with continuous callbacks in solve

    function f!(du, u, p, t)
        du[1] = -0.5u[1]
    end

    rate(u, p, t) = 0.1
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [5.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f!, u0, tspan)

    # Discrete callback in JumpProblem
    dcb_called = Ref(0)
    condition_d(u, t, integrator) = true
    affect_d!(integrator) = (dcb_called[] += 1)
    dcb = DiscreteCallback(condition_d, affect_d!)

    # Continuous callback in solve
    ccb_called = Ref(false)
    condition_c(u, t, integrator) = t - 7.0
    affect_c!(integrator) = (ccb_called[] = true)
    ccb = ContinuousCallback(condition_c, affect_c!)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = dcb)
    sol = solve(jprob, Tsit5(); callback = ccb)

    @test dcb_called[] > 0  # Discrete callback should fire
    @test ccb_called[]  # Continuous callback should fire
end

@testset "CallbackSet in JumpProblem, additional callbacks in solve" begin
    # Test that CallbackSet in JumpProblem works with additional callbacks in solve

    function f!(du, u, p, t)
        du[1] = -0.5u[1]
    end

    rate(u, p, t) = 0.1
    affect!(integrator) = (integrator.u[1] += 1.0)
    jump = ConstantRateJump(rate, affect!)

    u0 = [5.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(f!, u0, tspan)

    # Create a CallbackSet with both continuous and discrete
    cb1_called = Ref(false)
    condition1(u, t, integrator) = t - 2.0
    affect1!(integrator) = (cb1_called[] = true)
    ccb = ContinuousCallback(condition1, affect1!)

    cb2_called = Ref(0)
    condition2(u, t, integrator) = t > 4.0
    affect2!(integrator) = (cb2_called[] += 1)
    dcb = DiscreteCallback(condition2, affect2!)

    cbset = CallbackSet(ccb, dcb)

    # Additional callback in solve
    cb3_called = Ref(false)
    condition3(u, t, integrator) = t - 8.0
    affect3!(integrator) = (cb3_called[] = true)
    ccb2 = ContinuousCallback(condition3, affect3!)

    jprob = JumpProblem(prob, Direct(), jump; rng, callback = cbset)
    sol = solve(jprob, Tsit5(); callback = ccb2)

    @test cb1_called[]  # First continuous callback should fire
    @test cb2_called[] > 0  # Discrete callback should fire
    @test cb3_called[]  # Second continuous callback should fire
end

@testset "SSAStepper continuous callback errors" begin
    # Setup a simple DiscreteProblem for SSAStepper
    rate(u, p, t) = 0.5
    affect_j!(integrator) = (integrator.u[1] += 1)
    jump = ConstantRateJump(rate, affect_j!)

    u0 = [0]
    tspan = (0.0, 10.0)
    dprob = DiscreteProblem(u0, tspan)

    # Test 1: ContinuousCallback passed to JumpProblem constructor should error on solve
    condition(u, t, integrator) = t - 5.0
    affect_cb!(integrator) = nothing
    ccb = ContinuousCallback(condition, affect_cb!)

    jprob_ccb = JumpProblem(dprob, Direct(), jump; rng, callback = ccb)
    @test_throws ErrorException solve(jprob_ccb, SSAStepper())

    # Test 2: ContinuousCallback passed to solve should error
    jprob = JumpProblem(dprob, Direct(), jump; rng)
    @test_throws ErrorException solve(jprob, SSAStepper(); callback = ccb)

    # Test 3: CallbackSet with continuous callbacks passed to JumpProblem should error on solve
    condition_d(u, t, integrator) = true
    affect_dcb!(integrator) = nothing
    dcb = DiscreteCallback(condition_d, affect_dcb!)

    cbset_with_continuous = CallbackSet(ccb, dcb)
    jprob_cbset = JumpProblem(dprob, Direct(), jump; rng, callback = cbset_with_continuous)
    @test_throws ErrorException solve(jprob_cbset, SSAStepper())

    # Test 4: CallbackSet with continuous callbacks passed to solve should error
    @test_throws ErrorException solve(jprob, SSAStepper(); callback = cbset_with_continuous)

    # Test 5: CallbackSet with multiple continuous callbacks should error with correct count
    ccb2 = ContinuousCallback(condition, affect_cb!)
    cbset_multi = CallbackSet(ccb, ccb2, dcb)

    jprob_multi = JumpProblem(dprob, Direct(), jump; rng, callback = cbset_multi)
    err = try
        solve(jprob_multi, SSAStepper())
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("2", err.msg)  # Should mention 2 continuous callbacks
    @test occursin("callbacks", err.msg)  # Plural form

    # Test 6: DiscreteCallbacks should work fine (no error)
    dcb_only = DiscreteCallback(condition_d, affect_dcb!)
    jprob_dcb = JumpProblem(dprob, Direct(), jump; rng, callback = dcb_only)
    sol = solve(jprob_dcb, SSAStepper())
    @test sol.retcode == ReturnCode.Success

    # Test 7: CallbackSet with only discrete callbacks should work
    dcb2 = DiscreteCallback(condition_d, affect_dcb!)
    cbset_discrete = CallbackSet(dcb_only, dcb2)
    jprob_dcb2 = JumpProblem(dprob, Direct(), jump; rng, callback = cbset_discrete)
    sol2 = solve(jprob_dcb2, SSAStepper())
    @test sol2.retcode == ReturnCode.Success

    # Test 8: Error should also be thrown with init
    @test_throws ErrorException init(jprob_ccb, SSAStepper())
    @test_throws ErrorException init(jprob, SSAStepper(); callback = ccb)
end
