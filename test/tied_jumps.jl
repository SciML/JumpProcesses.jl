using JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Random, SciMLBase, StableRNGs,
      Statistics, Test

# An RNG whose exponential draws are all 1 and uniform draws 0.25, so the jump times are
# deterministic
struct TiedJumpsRNG <: AbstractRNG end
Random.randexp(::TiedJumpsRNG) = 1.0
Random.rand(::TiedJumpsRNG, ::Random.SamplerTrivial{Random.CloseOpen01{Float64}}) = 0.25

zero_drift!(du, u, p, t) = (fill!(du, 0); nothing)

# Jump times that round to the current time in ODE and SDE problems. Each A, at rate 1, sets
# X = 1, and B, X → Y at rate 1e6 X, then fires within about 1e-6, which rounds to A's time
# for t ≥ 1e12. B's stop used to be at the current time, which the integrator never reaches,
# so no jump fired after the first A. 1e6 + 1 is exact, so a total rate updated
# incrementally keeps A's rate. A larger rate for B would also make `RDirect` slow, since it
# rejects candidates against the largest rate seen until it updates its maximum.
@testset "Jumps tied with the current time, all aggregators" begin
    rateA(u, p, t) = 1.0
    affectA!(integrator) = (integrator.u[1] += 1; nothing)
    rateB(u, p, t) = 1e6 * u[1]
    affectB!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    jumps = (ConstantRateJump(rateA, affectA!), ConstantRateJump(rateB, affectB!))
    dep_graph = [[1, 2], [2]]
    vartojumps_map = [[2], Int[]]
    jumptovars_map = [[1], [1, 2]]
    tspan = (1e12, 1e12 + 10)
    # adaptive SDE steps can stop with `DtLessThanMin` near 1e12, where stops an ulp apart
    # force steps of one ulp
    for (prob, alg, kwargs) in (
            (ODEProblem(zero_drift!, [0.0, 0.0], tspan), Tsit5(), (;)),
            (SDEProblem(zero_drift!, zero_drift!, [0.0, 0.0], tspan), SRIW1(),
            (; adaptive = false, dt = 0.25))),
        aggregator in JumpProcesses.JUMP_AGGREGATORS

        jprob = JumpProblem(prob, aggregator, jumps...; dep_graph, vartojumps_map,
            jumptovars_map, save_positions = (false, false))
        rng = StableRNG(12345)
        sols = [solve(jprob, alg; rng, save_everystep = false, kwargs...) for _ in 1:200]
        @test all(sol -> SciMLBase.successful_retcode(sol), sols)
        @test all(sol -> sol.t[end] == tspan[2], sols)
        # every A was followed by its B, and A fired at rate 1 throughout
        @test all(sol -> sol.u[end][1] == 0, sols)
        @test isapprox(mean(sol -> sol.u[end][2], sols), 10; rtol = 0.1)
    end
end

# A sets u = 1, which switches on B at rate `rate_b`; B sets u = 2. With TiedJumpsRNG, A
# fires at 1, and B, at rate 1e20, at the same time.
function two_jump_problem(affectA!, prob; rate_b = 1e20, save_positions = (false, true))
    a = ConstantRateJump((u, p, t) -> u[1] == 0 ? 1.0 : 0.0, affectA!)
    times_b = Float64[]
    affectB!(integrator) = (integrator.u[1] = 2; push!(times_b, integrator.t); nothing)
    b = ConstantRateJump((u, p, t) -> u[1] == 1 ? rate_b : 0.0, affectB!)
    return JumpProblem(prob, Direct(), a, b; save_positions), times_b
end

function solve_two_jumps(affectA!; kwargs...)
    prob = ODEProblem(zero_drift!, [0.0], (0.0, 2.0))
    jprob, times_b = two_jump_problem(affectA!, prob; kwargs...)
    sol = solve(jprob, Tsit5(); dt = 1.0, adaptive = false, save_everystep = false,
        rng = TiedJumpsRNG())
    return sol, times_b
end

@testset "Saves between tied jumps in ODE problems" begin
    # as for separate events at the same time: after A, and after B
    sol, times_b = solve_two_jumps(integrator -> (integrator.u[1] = 1; nothing))
    @test times_b == [1.0]
    @test only.(sol.u[sol.t .== 1.0]) == [1.0, 2.0]
    # before and after each
    sol, times_b = solve_two_jumps(integrator -> (integrator.u[1] = 1; nothing);
        save_positions = (true, true))
    @test only.(sol.u[sol.t .== 1.0]) == [0.0, 1.0, 1.0, 2.0]
end

# `terminate!` may set any retcode
@testset "No tied jump after terminate!" begin
    for terminate_with! in (terminate!, i -> terminate!(i, ReturnCode.Success))
        affectA!(integrator) = (integrator.u[1] = 1; terminate_with!(integrator); nothing)
        sol, times_b = solve_two_jumps(affectA!)
        @test isempty(times_b)
        @test only(sol.u[end]) == 1

        # SSAStepper too
        jprob, times_b = two_jump_problem(affectA!, DiscreteProblem([0.0], (0.0, 2.0)))
        sol = solve(jprob, SSAStepper(); rng = TiedJumpsRNG())
        @test isempty(times_b)
        @test only(sol.u[end]) == 1
    end
end

# The next jump's stop used to be registered after the affect, which restarted an ODE solve
# that the affect terminated with a retcode other than `Terminated`
@testset "A jump's terminate! stops an ODE solve, whatever the retcode" begin
    times = Float64[]
    function affect!(integrator)
        push!(times, integrator.t)
        terminate!(integrator, ReturnCode.Success)
        nothing
    end
    jprob = JumpProblem(ODEProblem(zero_drift!, [0.0], (0.0, 10.0)), Direct(),
        ConstantRateJump((u, p, t) -> 1.0, affect!))
    sol = solve(jprob, Tsit5(); rng = TiedJumpsRNG())
    @test times == [1.0]
    @test sol.t[end] == 1.0
end

@testset "Next jump scheduled from the time after an affect that changes it" begin
    function affectA!(integrator)
        SciMLBase.change_t_via_interpolation!(integrator, 0.8)
        integrator.u[1] = 1
        nothing
    end
    _, times_b = solve_two_jumps(affectA!; rate_b = 2.0, save_positions = (false, false))
    @test times_b ≈ [1.3]
end

# A fires at 1 and at the end time, 2, where it switches itself off and switches on B, whose
# time rounds to 2. B's time is really past the end, and DirectCR doesn't select a next jump
# there, so nothing more may fire.
@testset "No tied jumps at the end time" begin
    for aggregator in (Direct(), DirectCR(), NRM())
        calls = Symbol[]
        record!(integrator, name) = (push!(calls, name); integrator.u[1] += 1; nothing)
        a = ConstantRateJump((u, p, t) -> u[1] < 2 ? 1.0 : 0.0, i -> record!(i, :A))
        b = ConstantRateJump((u, p, t) -> u[1] == 2 ? 1e20 : 0.0, i -> record!(i, :B))
        jprob = JumpProblem(ODEProblem(zero_drift!, [0.0], (0.0, 2.0)), aggregator, a, b;
            dep_graph = [[1, 2], [1, 2]])
        solve(jprob, Tsit5(); dt = 0.25, adaptive = false, rng = TiedJumpsRNG())
        @test calls == [:A, :A]
    end
end

# A fires at 1, when Coevolve also renews B's `rateinterval`, a tied candidate that is
# rejected
function jump_and_tied_rejection()
    a = ConstantRateJump((u, p, t) -> iszero(u[1]) ? 1.0 : 0.0, i -> (i.u[1] = 1; nothing))
    b = VariableRateJump((u, p, t) -> 0.0, i -> error("zero-rate jump fired");
        urate = (u, p, t) -> 0.0, rateinterval = (u, p, t) -> 1.0)
    return a, b
end

# A switches on the ODE's slope. The rejection used to clear A's derivative discontinuity, so
# Tsit5 kept the old slope.
@testset "Coevolve rejection after a tied jump keeps its derivative discontinuity" begin
    drift!(du, u, p, t) = (du[1] = 0; du[2] = u[1]; nothing)
    jprob = JumpProblem(ODEProblem(drift!, zeros(2), (0.0, 2.0)), Coevolve(),
        jump_and_tied_rejection()...; dep_graph = [[1], [2]])
    sol = solve(jprob, Tsit5(); dt = 1.0, adaptive = false, rng = TiedJumpsRNG())
    @test sol.u[end][2] ≈ 1.0
end

# The rejection used to make SSAStepper skip the save after A
@testset "SSAStepper saves a jump followed by a rejected tied candidate" begin
    jprob = JumpProblem(DiscreteProblem([0.0], (0.0, 2.0)), Coevolve(),
        jump_and_tied_rejection()...; dep_graph = [[1], [2]])
    sol = solve(jprob, SSAStepper(); rng = TiedJumpsRNG())
    @test sol.t == [0.0, 1.0, 2.0]
    @test only.(sol.u) == [0.0, 1.0, 1.0]
end

# A switches on the ODE's slope at 1, and B, tied with it, only does bookkeeping and says so
# with `derivative_discontinuity!(integrator, false)`. B used to clear A's derivative
# discontinuity, so Tsit5 kept the old slope.
@testset "Derivative discontinuity of any tied jump restarts the solver" begin
    function drift!(du, u, p, t)
        du[1] = 0
        du[2] = u[1] >= 1 ? 1 : 0
        nothing
    end
    a = ConstantRateJump((u, p, t) -> u[1] == 0 ? 1.0 : 0.0,
        integrator -> (integrator.u[1] = 1; nothing))
    function affectB!(integrator)
        integrator.u[1] = 2
        # the slope is 1 both before and after B
        SciMLBase.derivative_discontinuity!(integrator, false)
        nothing
    end
    b = ConstantRateJump((u, p, t) -> u[1] == 1 ? 1e20 : 0.0, affectB!)
    for aggregator in (Direct(), Coevolve())
        jprob = JumpProblem(ODEProblem(drift!, zeros(2), (0.0, 2.0)), aggregator, a, b;
            dep_graph = [[1, 2], [2]], save_positions = (false, false))
        sol = solve(jprob, Tsit5(); dt = 1.0, adaptive = false, save_everystep = false,
            rng = TiedJumpsRNG())
        @test sol.u[end][2] ≈ 1.0
    end
end

# With a retcode other than `Terminated`, SSAStepper used to advance to the end time after the
# affect and run the callbacks there
@testset "SSAStepper stops at terminate!, whatever the retcode" begin
    endpoint_calls = Ref(0)
    endpoint = DiscreteCallback((u, t, integrator) -> t == 2.0,
        integrator -> (endpoint_calls[] += 1; integrator.u[1] += 100; nothing))
    function affect!(integrator)
        integrator.u[1] = 1
        terminate!(integrator, ReturnCode.Success)
        nothing
    end
    jprob = JumpProblem(DiscreteProblem([0.0], (0.0, 2.0)), Direct(),
        ConstantRateJump((u, p, t) -> 1.0, affect!))
    sol = solve(jprob, SSAStepper(); callback = endpoint, rng = TiedJumpsRNG())
    @test sol.t[end] == 1.0
    @test endpoint_calls[] == 0
    @test only(sol.u[end]) == 1
end
