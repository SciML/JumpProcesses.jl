using JumpProcesses, OrdinaryDiffEq, Random, StableRNGs, Test

# 2A -> 0 from A = 11 leaves one A, and every rate is zero. RSSA and RSSACR draw candidates
# from bounds on the rates, and the bound of 2A -> 0 stays positive while A is inside its
# bracket, so they rejected every candidate, whatever the end time. CCNRM gives a zero rate
# the time floatmax, which it reached with an infinite end time.
@testset "SSAs stop when every rate is zero" begin
    maj = MassActionJump([1.0], [[1 => 2]], [[1 => -2]])
    for tend in (100.0, Inf)
        for agg in JumpProcesses.JUMP_AGGREGATORS
            jprob = JumpProblem(DiscreteProblem([11], (0.0, tend)), agg, maj)
            sol = solve(jprob, SSAStepper(); rng = StableRNG(1))
            @test sol.u[end] == [1]
            @test sol.t[end] == tend
        end
    end
end

# With 0 -> B at rate 1e-30 the total rate is positive, but RSSA and RSSACR would need about
# 36 / 1e-30 candidates to accept one; once a candidate comes after the end time, so does
# the next jump.
@testset "SSAs stop at the end time when every rate is tiny" begin
    maj = MassActionJump([1.0, 1e-30], [[1 => 2], Pair{Int, Int}[]], [[1 => -2], [2 => 1]])
    for agg in JumpProcesses.JUMP_AGGREGATORS
        jprob = JumpProblem(DiscreteProblem([11, 0], (0.0, 10.0)), agg, maj)
        sol = solve(jprob, SSAStepper(); rng = StableRNG(1))
        @test sol.u[end] == [1, 0]
        @test sol.t[end] == 10.0
    end
end

# An RNG whose uniform draws are all 0.1 and exponential draws 1, so the jump times are
# deterministic
struct ZeroRatesRNG <: AbstractRNG end
Random.randexp(::ZeroRatesRNG) = 1.0
Random.rand(::ZeroRatesRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64}}) = 0.1

# An ODE or SDE integrator executes the next jump when it reaches the jump's time, the end
# time included. RSSA and RSSACR, after 16 rejected candidates in a row, gave the last one's
# time as the next jump time if it came at or after the end time, so at exactly the end time
# the last executed jump fired again. Here 2A -> 0 fires once from A = 3, every candidate
# after it is rejected, and the second solve ends at the 16th candidate's time.
@testset "RSSA and RSSACR fire no rejected candidate at the end of an ODE solve" begin
    maj = MassActionJump([1.0], [[1 => 2]], [[1 => -2]])
    zero_drift!(du, u, p, t) = (fill!(du, 0); nothing)
    for agg in (RSSA(), RSSACR())
        jprob = JumpProblem(ODEProblem(zero_drift!, [3.0], (0.0, 100.0)), agg, maj)
        sol = solve(jprob, Tsit5(); rng = ZeroRatesRNG())
        @test sol.u[end] == [1.0]
        tjump = sol.t[findfirst(==([1.0]), sol.u)]
        tend = tjump + 16 / jprob.discrete_jump_aggregation.sum_rate
        jprob = JumpProblem(ODEProblem(zero_drift!, [3.0], (0.0, tend)), agg, maj)
        sol = solve(jprob, Tsit5(); rng = ZeroRatesRNG())
        @test sol.u[end] == [1.0]
        @test sol.t[end] == tend
    end
end
