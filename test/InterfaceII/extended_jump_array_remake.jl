# Tests for JumpProblem remake with VariableRateJumps (ExtendedJumpArray case)
# This tests the fix for symbolic u0 with ExtendedJumpArray

using JumpProcesses, OrdinaryDiffEq, Test, SymbolicIndexingInterface
using StableRNGs

@testset "remake JumpProblem with VariableRateJumps (ExtendedJumpArray)" begin
    rng = StableRNG(12345)

    # Setup: Create an ODEProblem with SymbolCache for symbolic indexing
    f(du, u, p, t) = (du .= 0; nothing)
    g = ODEFunction(f; sys = SymbolCache([:X, :Y], [:k1, :k2], :t))
    oprob = ODEProblem(g, [10.0, 5.0], (0.0, 10.0), [1.0, 2.0])

    # Add a VariableRateJump to trigger ExtendedJumpArray
    vr_rate(u, p, t) = p[1] * u[1]
    vr_affect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
    vrj = VariableRateJump(vr_rate, vr_affect!)

    jprob = JumpProblem(oprob, vrj; rng)

    # Verify we have ExtendedJumpArray
    @test jprob.prob.u0 isa ExtendedJumpArray
    @test jprob.prob.u0.u == [10.0, 5.0]

    @testset "remake with numeric Vector{Float64}" begin
        original_jump_u = copy(jprob.prob.u0.jump_u)

        prob2 = remake(jprob; u0 = [20.0, 10.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u == [20.0, 10.0]
        # jump_u should be resampled (different from original)
        @test prob2.prob.u0.jump_u != original_jump_u
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
        original_jump_u = copy(jprob.prob.u0.jump_u)

        # This was the FAILING case - should work after fix
        prob2 = remake(jprob; u0 = [:X => 25.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u[1] == 25.0
        # jump_u should be resampled
        @test prob2.prob.u0.jump_u != original_jump_u
    end

    @testset "remake with multiple Symbol pairs" begin
        original_jump_u = copy(jprob.prob.u0.jump_u)

        prob2 = remake(jprob; u0 = [:X => 35.0, :Y => 15.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u == [35.0, 15.0]
        # jump_u should be resampled
        @test prob2.prob.u0.jump_u != original_jump_u
    end

    @testset "remake with Dict" begin
        original_jump_u = copy(jprob.prob.u0.jump_u)

        prob2 = remake(jprob; u0 = Dict(:X => 40.0))
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u[1] == 40.0
        # jump_u should be resampled
        @test prob2.prob.u0.jump_u != original_jump_u
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
        original_jump_u = copy(jprob.prob.u0.jump_u)

        prob2 = remake(jprob; u0 = [:X => 50.0], p = [:k1 => 3.0])
        @test prob2.prob.u0 isa ExtendedJumpArray
        @test prob2.prob.u0.u[1] == 50.0
        @test prob2.prob.p[1] == 3.0
        # jump_u should be resampled
        @test prob2.prob.u0.jump_u != original_jump_u
    end

    @testset "remake preserves problem solvability" begin
        # Ensure remade problems can actually be solved
        prob2 = remake(jprob; u0 = [5.0, 2.0])
        sol = solve(prob2, Tsit5())
        @test SciMLBase.successful_retcode(sol)

        # With symbolic map (after fix)
        prob3 = remake(jprob; u0 = [:X => 8.0])
        sol3 = solve(prob3, Tsit5())
        @test SciMLBase.successful_retcode(sol3)
    end
end
