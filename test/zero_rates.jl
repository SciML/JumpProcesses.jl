using JumpProcesses, StableRNGs, Test

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
