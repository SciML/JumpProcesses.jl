# Tests for JumpProblem remake with VariableRateJumps (ExtendedJumpArray case)
# This tests the fix for symbolic u0 with ExtendedJumpArray

using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, SciMLBase, Test,
      SymbolicIndexingInterface
using StableRNGs

@testset "remake JumpProblem with VariableRateJumps (ExtendedJumpArray)" begin
    # Setup: Create an ODEProblem with SymbolCache for symbolic indexing
    f(du, u, p, t) = (du .= 0; nothing)
    g = ODEFunction(f; sys = SymbolCache([:X, :Y], [:k1, :k2], :t))
    oprob = ODEProblem(g, [10.0, 5.0], (0.0, 10.0), [1.0, 2.0])

    # Add a VariableRateJump to trigger ExtendedJumpArray
    vr_rate(u, p, t) = p[1] * u[1]
    vr_affect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
    vrj = VariableRateJump(vr_rate, vr_affect!)

    jprob = JumpProblem(oprob, vrj)

    # Verify we have ExtendedJumpArray
    @test jprob.prob.u0 isa ExtendedJumpArray
    @test jprob.prob.u0.u == [10.0, 5.0]

    # Solve original problem and capture jump_u after initialization
    orig_integrator = init(jprob, Tsit5(); rng = StableRNG(42))
    orig_jump_u = copy(orig_integrator.u.jump_u)

    @testset "remake with numeric Vector{Float64}" begin
        prob2 = remake(jprob; u0 = [20.0, 10.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u == [20.0, 10.0]
        @test all(iszero, prob2.prob.u0.jump_u)
        # After init, callback sets fresh jump_u thresholds (different RNG seed)
        integrator = init(prob2, Tsit5(); rng = StableRNG(99))
        @test any(!iszero, integrator.u.jump_u)
        @test integrator.u.jump_u != orig_jump_u
    end

    @testset "remake with ExtendedJumpArray (no resample)" begin
        # User passes ExtendedJumpArray directly - should preserve jump_u
        new_u0 = deepcopy(jprob.prob.u0)
        new_u0.u[1] = 30.0
        original_jump_u = copy(new_u0.jump_u)

        prob2 = remake(jprob; u0 = new_u0)
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u[1] == 30.0
        # jump_u should NOT be resampled - user has full control
        @test prob2.prob.u0.jump_u == original_jump_u
    end

    @testset "remake with Symbol pairs" begin
        prob2 = remake(jprob; u0 = [:X => 25.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u[1] == 25.0
        @test all(iszero, prob2.prob.u0.jump_u)
        # After init, callback sets fresh jump_u thresholds (different RNG seed)
        integrator = init(prob2, Tsit5(); rng = StableRNG(99))
        @test any(!iszero, integrator.u.jump_u)
        @test integrator.u.jump_u != orig_jump_u
    end

    @testset "remake with multiple Symbol pairs" begin
        prob2 = remake(jprob; u0 = [:X => 35.0, :Y => 15.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u == [35.0, 15.0]
        @test all(iszero, prob2.prob.u0.jump_u)
        integrator = init(prob2, Tsit5(); rng = StableRNG(99))
        @test any(!iszero, integrator.u.jump_u)
        @test integrator.u.jump_u != orig_jump_u
    end

    @testset "remake with Dict" begin
        prob2 = remake(jprob; u0 = Dict(:X => 40.0))
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u[1] == 40.0
        @test all(iszero, prob2.prob.u0.jump_u)
        integrator = init(prob2, Tsit5(); rng = StableRNG(99))
        @test any(!iszero, integrator.u.jump_u)
        @test integrator.u.jump_u != orig_jump_u
    end

    @testset "remake with parameters only (u0 unchanged)" begin
        original_u0 = deepcopy(jprob.prob.u0)

        prob2 = remake(jprob; p = [:k1 => 5.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        # u0 should be unchanged (same reference or equal values)
        @test prob2.prob.u0.u == original_u0.u
        @test prob2.prob.u0.jump_u == original_u0.jump_u
    end

    @testset "remake with both u0 and p" begin
        prob2 = remake(jprob; u0 = [:X => 50.0], p = [:k1 => 3.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u[1] == 50.0
        @test prob2.prob.p[1] == 3.0
        @test all(iszero, prob2.prob.u0.jump_u)
        integrator = init(prob2, Tsit5(); rng = StableRNG(99))
        @test any(!iszero, integrator.u.jump_u)
        @test integrator.u.jump_u != orig_jump_u
    end

    @testset "remake preserves problem solvability" begin
        # Solve original, then remake and solve again — jump_u should differ
        sol1 = solve(jprob, Tsit5(); rng = StableRNG(42))
        @test SciMLBase.successful_retcode(sol1)

        prob2 = remake(jprob; u0 = [5.0, 2.0])
        sol2 = solve(prob2, Tsit5(); rng = StableRNG(99))
        @test SciMLBase.successful_retcode(sol2)
        # Different RNG seeds → different jump_u thresholds after init
        @test sol1.u[2].jump_u != sol2.u[2].jump_u

        # With symbolic map
        prob3 = remake(jprob; u0 = [:X => 8.0])
        sol3 = solve(prob3, Tsit5(); rng = StableRNG(77))
        @test SciMLBase.successful_retcode(sol3)
        @test sol1.u[2].jump_u != sol3.u[2].jump_u
    end
end

# `remake` zeroes `jump_u`, so every solve must draw new `VR_FRM` thresholds. A single jump
# with constant rate `λ` first fires at an Exp(λ) time `τ`, and the threshold saved at
# `t = 0` after callback initialization must be `jump_u = -λτ`. The first saved point at
# `t = 0` precedes callback initialization, so its `jump_u` is not a threshold.
@testset "VR_FRM draws new thresholds at every solve after remake" begin
    λ = 2.0
    N = 1000
    f!(du, u, p, t) = (du .= 0; nothing)
    g!(du, u, p, t) = (du[1] = 0.1; du[2] = 0; nothing)
    rate(u, p, t) = λ
    affect!(integrator) = (integrator.u[2] += 1; nothing)
    vrj = VariableRateJump(rate, affect!)

    # Returns the first jump time, and whether the solve succeeded and the threshold saved
    # at `t = 0` after initialization is nonzero and gives that time.
    function check_sol(sol)
        i0 = findlast(==(0.0), sol.t)
        threshold = -sol.u[i0].jump_u[1]
        i1 = findfirst(u -> u.u[2] > 0, sol.u)
        i1 === nothing && return (Inf, false)
        τ = sol.t[i1]
        τ, SciMLBase.successful_retcode(sol) && (sol.u[i1].u[2] == 1) &&
           (threshold > 0) && isapprox(τ, threshold / λ; rtol = 1e-6)
    end

    # Kolmogorov-Smirnov distance between the sampled times and Exp(λ)
    function ks_distance(τs)
        n = length(τs)
        F = 1 .- exp.(-λ .* sort(τs))
        maximum(max.((1:n) ./ n .- F, F .- (0:(n - 1)) ./ n))
    end
    ks_critical = 1.95 / sqrt(N)    # significance level 0.001

    function test_results(results)
        τs = first.(results)
        @test all(last, results)
        @test allunique(τs)
        @test ks_distance(τs) < ks_critical
    end

    tspan = (0.0, 10.0)
    cases = [("ODE", ODEProblem(f!, [1.0, 0.0], tspan), Tsit5(), (;)),
        ("SDE", SDEProblem(f!, g!, [1.0, 0.0], tspan), SOSRI(), (;)),
        ("SDE", SDEProblem(f!, g!, [1.0, 0.0], tspan), EM(), (; dt = 0.01))]
    for (name, prob, alg, kwargs) in cases
        @testset "$name, $(nameof(typeof(alg)))" begin
            jprob = JumpProblem(prob, vrj; vr_aggregator = VR_FRM())
            jprob2 = remake(jprob; u0 = [2.0, 0.0])
            @test all(iszero, jprob2.prob.u0.jump_u)
            # `remake` also zeroed the original problem's `jump_u`, which it aliases
            @test all(iszero, jprob.prob.u0.jump_u)
            rng = StableRNG(1234)

            @testset "solve remade problem" begin
                test_results([check_sol(solve(jprob2, alg; rng, kwargs...)) for _ in 1:N])
            end

            @testset "solve original problem" begin
                test_results([check_sol(solve(jprob, alg; rng, kwargs...)) for _ in 1:N])
            end

            @testset "reinit! integrator of remade problem" begin
                integrator = init(jprob2, alg; rng, kwargs...)
                results = map(1:N) do _
                    reinit!(integrator)
                    solve!(integrator)
                    check_sol(integrator.sol)
                end
                test_results(results)
            end
        end
    end
end
