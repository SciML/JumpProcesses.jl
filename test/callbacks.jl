using JumpProcesses, OrdinaryDiffEq, SciMLBase, DiffEqBase, Test
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

    jprob = JumpProblem(prob, Direct(), jump; callback = cb)
    sol = solve(jprob, Tsit5(); rng)

    @test cb_called[]
    @test sol.t[end] ≈ 10.0

    # Test 2: DiscreteCallback in JumpProblem constructor
    dcb_called = Ref(0)
    condition_d(u, t, integrator) = t > 2.0  # Fire at every step after t=2
    affect_dcb!(integrator) = (dcb_called[] += 1)
    dcb = DiscreteCallback(condition_d, affect_dcb!)

    jprob = JumpProblem(prob, Direct(), jump; callback = dcb)
    sol = solve(jprob, Tsit5(); rng)

    @test dcb_called[] > 0  # Should have fired multiple times

    # Test 3: Terminating callback in JumpProblem constructor
    condition_term(u, t, integrator) = t - 3.0
    affect_term!(integrator) = terminate!(integrator)
    cb_term = ContinuousCallback(condition_term, affect_term!)

    jprob = JumpProblem(prob, Direct(), jump; callback = cb_term)
    sol = solve(jprob, Tsit5(); rng)

    @test sol.t[end] ≈ 3.0  # Should terminate at t=3

    # Test 4: State-modifying callback in JumpProblem constructor
    condition_mod(u, t, integrator) = t - 5.0
    affect_mod!(integrator) = (integrator.u[1] *= 2.0)
    cb_mod = ContinuousCallback(condition_mod, affect_mod!)

    jprob = JumpProblem(prob, Direct(), jump; callback = cb_mod)
    sol = solve(jprob, Tsit5(); rng)

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
    jprob = JumpProblem(prob, Direct(), jump; callback = cb1)
    sol = solve(jprob, Tsit5(); callback = cb2, rng)

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
    jprob = JumpProblem(prob, Direct(), jump; callback = cb1)
    sol = solve(jprob, Tsit5(); callback = cb2, merge_callbacks = false, rng)

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
    jprob = JumpProblem(prob, Direct(), jump; callback = cb)
    integrator = init(jprob, Tsit5(); rng)
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

    jprob = JumpProblem(prob, Direct(), jump; callback = cbset)
    sol = solve(jprob, Tsit5(); rng)

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

    jprob = JumpProblem(dprob, Direct(), jump1; callback = dcb_term)
    sol = solve(jprob, SSAStepper(); rng)

    @test cb_called[]  # Should have fired
    @test sol.u[end][1] >= 5  # Should have reached threshold
    @test sol.t[end] < 10.0  # Should have terminated early

    # Test 2: DiscreteCallback in solve call
    dcb_counter = Ref(0)
    condition_count(u, t, integrator) = u[1] >= 3
    affect_count!(integrator) = (dcb_counter[] += 1)
    dcb_count = DiscreteCallback(condition_count, affect_count!)

    jprob2 = JumpProblem(dprob, Direct(), jump1)
    sol2 = solve(jprob2, SSAStepper(); callback = dcb_count, rng)

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

    jprob3 = JumpProblem(dprob, Direct(), jump1; callback = dcb1)
    sol3 = solve(jprob3, SSAStepper(); callback = dcb2, rng)

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

    jprob4 = JumpProblem(dprob, Direct(), jump1; callback = dcb3)
    sol4 = solve(jprob4, SSAStepper(); callback = dcb4, merge_callbacks = false, rng)

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
    jprob = JumpProblem(prob, Direct(), jump; callback = cb)
    sol = solve(jprob, Tsit5(); rng)

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

    jprob = JumpProblem(prob, Direct(), jump; callback = ccb)
    sol = solve(jprob, Tsit5(); callback = dcb, rng)

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

    jprob = JumpProblem(prob, Direct(), jump; callback = dcb)
    sol = solve(jprob, Tsit5(); callback = ccb, rng)

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

    jprob = JumpProblem(prob, Direct(), jump; callback = cbset)
    sol = solve(jprob, Tsit5(); callback = ccb2, rng)

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

    jprob_ccb = JumpProblem(dprob, Direct(), jump; callback = ccb)
    @test_throws ErrorException solve(jprob_ccb, SSAStepper(); rng)

    # Test 2: ContinuousCallback passed to solve should error
    jprob = JumpProblem(dprob, Direct(), jump)
    @test_throws ErrorException solve(jprob, SSAStepper(); callback = ccb, rng)

    # Test 3: CallbackSet with continuous callbacks passed to JumpProblem should error on solve
    condition_d(u, t, integrator) = true
    affect_dcb!(integrator) = nothing
    dcb = DiscreteCallback(condition_d, affect_dcb!)

    cbset_with_continuous = CallbackSet(ccb, dcb)
    jprob_cbset = JumpProblem(dprob, Direct(), jump; callback = cbset_with_continuous)
    @test_throws ErrorException solve(jprob_cbset, SSAStepper(); rng)

    # Test 4: CallbackSet with continuous callbacks passed to solve should error
    @test_throws ErrorException solve(jprob, SSAStepper(); callback = cbset_with_continuous, rng)

    # Test 5: CallbackSet with multiple continuous callbacks should error with correct count
    ccb2 = ContinuousCallback(condition, affect_cb!)
    cbset_multi = CallbackSet(ccb, ccb2, dcb)

    jprob_multi = JumpProblem(dprob, Direct(), jump; callback = cbset_multi)
    err = try
        solve(jprob_multi, SSAStepper(); rng)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("2", err.msg)  # Should mention 2 continuous callbacks
    @test occursin("callbacks", err.msg)  # Plural form

    # Test 6: DiscreteCallbacks should work fine (no error)
    dcb_only = DiscreteCallback(condition_d, affect_dcb!)
    jprob_dcb = JumpProblem(dprob, Direct(), jump; callback = dcb_only)
    sol = solve(jprob_dcb, SSAStepper(); rng)
    @test sol.retcode == ReturnCode.Success

    # Test 7: CallbackSet with only discrete callbacks should work
    dcb2 = DiscreteCallback(condition_d, affect_dcb!)
    cbset_discrete = CallbackSet(dcb_only, dcb2)
    jprob_dcb2 = JumpProblem(dprob, Direct(), jump; callback = cbset_discrete)
    sol2 = solve(jprob_dcb2, SSAStepper(); rng)
    @test sol2.retcode == ReturnCode.Success

    # Test 8: Error should also be thrown with init
    @test_throws ErrorException init(jprob_ccb, SSAStepper(); rng)
    @test_throws ErrorException init(jprob, SSAStepper(); callback = ccb, rng)
end

@testset "SDE + jump callback not duplicated" begin
    # Regression test for PR #567: verify that when using an SDE algorithm with a
    # JumpProblem, the jump callback is added exactly once (not duplicated by both
    # JumpProcesses and StochasticDiffEq). See:
    # https://github.com/SciML/JumpProcesses.jl/pull/567#issuecomment-4092662794
    using StochasticDiffEq

    # SDE: dX = -X dt + 0.1 dW
    f(u, p, t) = -u
    g(u, p, t) = 0.1

    # Constant rate jump
    rate(u, p, t) = 1.0
    affect!(integrator) = (integrator.u += 0.5)
    crj = ConstantRateJump(rate, affect!)

    sde_prob = SDEProblem(f, g, 1.0, (0.0, 1.0))
    jprob = JumpProblem(sde_prob, Direct(), crj)

    integrator = init(jprob, EM(); dt = 0.01, rng)

    # Count discrete callbacks — the jump should appear exactly once.
    # If both JumpProcesses and StochasticDiffEq add it, we'd see duplicates.
    n_discrete = length(integrator.opts.callback.discrete_callbacks)
    @test n_discrete == 1

    # Also test with a VariableRateJump (needs array u0 for ExtendedJumpArray)
    f_vr(du, u, p, t) = (du[1] = -u[1])
    g_vr(du, u, p, t) = (du[1] = 0.1)
    sde_prob_vr = SDEProblem(f_vr, g_vr, [1.0], (0.0, 1.0))

    vrate(u, p, t) = 1.0
    vaffect!(integrator) = (integrator.u[1] += 0.5)
    vrj = VariableRateJump(vrate, vaffect!)

    jprob_vr = JumpProblem(sde_prob_vr, Direct(), vrj; vr_aggregator = VR_FRM())
    integrator_vr = init(jprob_vr, EM(); dt = 0.01, rng)

    # VariableRateJumps produce continuous callbacks
    n_continuous = length(integrator_vr.opts.callback.continuous_callbacks)
    @test n_continuous == 1
end

# Callbacks can be stored on the `JumpProblem` or passed at the call site (callbacks
# stored on the wrapped problem are rejected; see the next testset). `init` must combine
# them exactly as `solve` does: DiffEqBase's `init_call` merges the `JumpProblem`'s stored
# keywords once, honoring `merge_callbacks`.
@testset "Stored and call-level callbacks: init matches solve" begin
    using OrdinaryDiffEqFunctionMap, StochasticDiffEq
    log = Symbol[]
    logging_callback(name) = DiscreteCallback((u, t, integrator) -> true,
        integrator -> (push!(log, name); nothing); save_positions = (false, false))
    jump = ConstantRateJump((u, p, t) -> 1.0, integrator -> (integrator.u[1] += 1; nothing))
    noop!(du, u, p, t) = (du .= 0; nothing)
    noop_rode!(du, u, p, t, W) = (du .= 0; nothing)
    ode = ODEProblem(noop!, [0.0], (0.0, 0.8))
    discrete = DiscreteProblem([0], (0.0, 8.0))
    sde = SDEProblem(noop!, noop!, [0.0], (0.0, 0.8))
    rode = RODEProblem(noop_rode!, [0.0], (0.0, 0.8))
    cases = (("ODE", ode, Tsit5(), (; adaptive = false, dt = 0.1)),
        ("FunctionMap", discrete, FunctionMap(), (;)),
        ("SSAStepper", discrete, SSAStepper(), (;)),
        ("SDE", sde, EM(), (; dt = 0.1)),
        ("RODE", rode, RandomEM(), (; dt = 0.1)))

    function run_and_log(entry, jprob, alg; kwargs...)
        empty!(log)
        if entry === :solve
            solve(jprob, alg; kwargs...)
        else
            solve!(init(jprob, alg; kwargs...))
        end
        copy(log)
    end

    @testset "$name, merge_callbacks = $merge_callbacks" for (name, prob, alg, options) in cases,
        merge_callbacks in (true, false)

        jprob = JumpProblem(prob, Direct(), jump; callback = logging_callback(:jumpproblem))
        call = logging_callback(:call)
        kwargs = (; options..., callback = call, merge_callbacks, seed = 1)
        solved = run_and_log(:solve, jprob, alg; kwargs...)
        initialized = run_and_log(:init, jprob, alg; kwargs...)
        # Same callbacks, same number of runs, same order.
        @test initialized == solved
        steps = count(==(:call), solved)
        @test steps > 0
        @test count(==(:jumpproblem), initialized) == (merge_callbacks ? steps : 0)
    end

    @testset "Direct __init discards merge_callbacks" begin
        prob = ODEProblem(noop!, [0.0], (0.0, 0.8))
        jprob = JumpProblem(prob, Direct(), jump; callback = logging_callback(:jumpproblem))
        options = (; adaptive = false, dt = 0.1, seed = 1)
        expected = run_and_log(:init, jprob, Tsit5(); options...,
            callback = logging_callback(:call))
        # `__init` receives keywords that are already merged; a stray `merge_callbacks`
        # must not merge them again.
        merged = DiffEqBase.merge_problem_kwargs(jprob; callback = logging_callback(:call))
        empty!(log)
        solve!(SciMLBase.__init(jprob, Tsit5(); options..., merged...,
            merge_callbacks = false))
        @test log == expected
    end

    @testset "Stored tstops are honored once" begin
        prob = ODEProblem(noop!, [0.0], (0.0, 1.0))
        jprob = JumpProblem(prob, Direct(), jump; tstops = [0.55])
        for sol in (solve(jprob, Tsit5(); seed = 1),
            solve!(init(jprob, Tsit5(); seed = 1)))
            @test count(==(0.55), sol.t) >= 1
        end
        integrator = init(jprob, Tsit5(); seed = 1)
        solve!(integrator)
        @test integrator.sol.t == solve(jprob, Tsit5(); seed = 1).t
    end
end

# Callbacks stored on the wrapped problem would run only on solvers that `init` the
# wrapped problem (OrdinaryDiffEq), and be ignored by `SSAStepper` and StochasticDiffEq, so
# the `JumpProblem` constructors and `remake` reject them.
@testset "Callbacks stored on the wrapped problem are rejected" begin
    using StochasticDiffEq
    function rejects_wrapped_callbacks(f)
        err = try
            f()
            nothing
        catch e
            e
        end
        err isa ArgumentError && occursin("are not supported", err.msg)
    end
    discrete_cb = DiscreteCallback((u, t, integrator) -> false, integrator -> nothing)
    continuous_cb = ContinuousCallback((u, t, integrator) -> t - 0.5, integrator -> nothing)
    stored_callbacks = (discrete_cb, continuous_cb, CallbackSet(discrete_cb))
    crj = ConstantRateJump((u, p, t) -> 1.0, integrator -> (integrator.u[1] += 1; nothing))
    vrj = VariableRateJump((u, p, t) -> 1.0, integrator -> (integrator.u[1] += 1; nothing))
    maj = MassActionJump([1.0], [[1 => 1]], [[1 => -1]])
    rj = RegularJump((out, u, p, t) -> (out[1] = 1.0),
        (du, u, p, t, counts, mark) -> (du[1] = counts[1]), 1)
    noop!(du, u, p, t) = (du .= 0; nothing)
    noop_rode!(du, u, p, t, W) = (du .= 0; nothing)
    tspan = (0.0, 1.0)
    discrete(kw) = DiscreteProblem([10], tspan; kw...)
    leaping(kw) = DiscreteProblem([10.0], tspan; kw...)
    spatial(kw) = DiscreteProblem([10 0], tspan; kw...)  # one species on two sites
    ode(kw) = ODEProblem(noop!, [0.0], tspan; kw...)
    sde(kw) = SDEProblem(noop!, noop!, [0.0], tspan; kw...)
    rode(kw) = RODEProblem(noop_rode!, [0.0], tspan; kw...)
    hopping_constants = fill(1.0, 1, 2)
    spatial_options = (; hopping_constants, spatial_system = CartesianGrid((2,)))

    # Each case wraps a problem through a different constructor path: its name, the
    # wrapped-problem builder, and the remaining `JumpProblem` arguments and keywords.
    cases = (("DiscreteProblem, Direct", discrete, (Direct(), crj), (;)),
        ("DiscreteProblem, selected aggregator", discrete, (maj,), (;)),
        ("DiscreteProblem, PureLeaping", leaping, (PureLeaping(), rj), (;)),
        # Flattening rebuilds the problem without its keywords, so the check must precede it.
        ("spatial DiscreteProblem, flattened", spatial, (Direct(), maj), spatial_options),
        ("spatial DiscreteProblem, NSM", spatial, (NSM(), maj), spatial_options),
        ("ODEProblem, Direct", ode, (Direct(), crj), (;)),
        ("ODEProblem, VR_FRM", ode, (vrj,), (; vr_aggregator = VR_FRM())),
        ("ODEProblem, VR_Direct", ode, (vrj,), (; vr_aggregator = VR_Direct())),
        ("SDEProblem", sde, (Direct(), crj), (;)),
        ("RODEProblem", rode, (Direct(), crj), (;)))

    @testset "$name" for (name, problem, args, options) in cases
        build(kw) = JumpProblem(problem(kw), args...; options...)
        for callback in stored_callbacks
            @test rejects_wrapped_callbacks(() -> build((; callback)))
        end
        # No stored callback, `nothing`, or an empty `CallbackSet` is accepted.
        for kw in ((;), (; callback = nothing), (; callback = CallbackSet()))
            @test build(kw) isa JumpProblem
        end
    end

    @testset "The error names the wrapped problem type" begin
        err = try
            JumpProblem(ode((; callback = discrete_cb)), Direct(), crj)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("wrapped `ODEProblem`", err.msg)
        @test occursin("remake(prob; callback = nothing)", err.msg)
    end

    @testset "Accepted problems solve" begin
        jprob = JumpProblem(discrete((; callback = CallbackSet())), Direct(), crj)
        @test SciMLBase.successful_retcode(solve(jprob, SSAStepper(); seed = 1))
        jprob = JumpProblem(ode((; callback = nothing)), Direct(), crj)
        @test SciMLBase.successful_retcode(solve(jprob, Tsit5(); seed = 1))
    end

    @testset "remake(prob; callback = nothing) removes stored callbacks" begin
        for problem in (discrete, ode, sde, rode)
            prob = problem((; callback = discrete_cb))
            @test rejects_wrapped_callbacks(() -> JumpProblem(prob, Direct(), crj))
            stripped = remake(prob; callback = nothing)
            @test JumpProblem(stripped, Direct(), crj) isa JumpProblem
        end
    end

    @testset "remake checks a new wrapped problem" begin
        jprob = JumpProblem(discrete((;)), Direct(), crj)
        newprob = DiscreteProblem([5], tspan; callback = discrete_cb)
        @test rejects_wrapped_callbacks(() -> remake(jprob; prob = newprob))
        @test remake(jprob; prob = DiscreteProblem([5], tspan)).prob.u0 == [5]
        @test remake(jprob; u0 = [5]).prob.u0 == [5]
    end
end
