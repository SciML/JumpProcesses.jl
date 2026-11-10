using JumpProcesses, StochasticDiffEq, SciMLBase, Test
using StableRNGs

# Jump-diffusion and tau-leaping solves through StochasticDiffEq's solver libraries: jumps
# draw from the integrator's RNG, `alias_jumps = false` copies the jump state, and
# mass-action rates follow the current parameters.

f!(du, u, p, t) = (du .= 0; nothing)
g!(du, u, p, t) = (du .= 0.1; nothing)
no_noise!(du, u, p, t) = (du .= 0; nothing)
birth_rate(u, p, t) = p[1]
birth!(integrator) = (integrator.u[1] += 1; nothing)

const T_END = 2.0

function birth_problem()
    sde = SDEProblem(f!, g!, [0.0], (0.0, T_END), [5.0])
    JumpProblem(sde, Direct(), ConstantRateJump(birth_rate, birth!))
end

same_path(sol1, sol2) = sol1.t == sol2.t && sol1.u == sol2.u
completed(sol) = SciMLBase.successful_retcode(sol) && sol.t[end] == T_END

# Two solves given equal RNGs, or the same seed, run to the end and follow the same path.
# The options are built for each solve, so the two solves never share an RNG object.
function check_replay(solution)
    for options in (() -> (; rng = StableRNG(1)), () -> (; seed = 7))
        sol1 = solution(; options()...)
        sol2 = solution(; options()...)
        @test completed(sol1)
        @test completed(sol2)
        @test same_path(sol1, sol2)
    end
end

@testset "Jump-diffusion solves replay with the same RNG" begin
    # EM, SOSRI, and ImplicitEM come from different StochasticDiffEq solver libraries.
    @testset "$(nameof(typeof(alg)))" for (alg, options) in (
        (EM(), (; dt = 0.01)), (SOSRI(), (;)), (ImplicitEM(), (; dt = 0.01)))
        jprob = birth_problem()
        solution(; kwargs...) = solve(jprob, alg; options..., kwargs...)
        check_replay(solution)
        @test !same_path(solution(; rng = StableRNG(7)), solution(; rng = StableRNG(8)))
    end

    @testset "TauLeaping" begin
        rate!(out, u, p, t) = (out[1] = p[1]; nothing)
        change!(du, u, p, t, counts, mark) = (du[1] = counts[1]; nothing)
        jprob = JumpProblem(DiscreteProblem([0.0], (0.0, T_END), [5.0]), Direct(),
            RegularJump(rate!, change!, 1))
        solution(; kwargs...) = solve(jprob, TauLeaping(); kwargs...)
        check_replay(solution)
    end
end

@testset "Copied jump state isolates integrators alive at the same time" begin
    jprob = birth_problem()
    copy_state = SciMLBase.SDEAliasSpecifier(; alias_jumps = false)
    expected1 = solve(jprob, EM(); dt = 0.01, rng = StableRNG(1), alias = copy_state)
    expected2 = solve(jprob, EM(); dt = 0.01, rng = StableRNG(2), alias = copy_state)
    integrator1 = init(jprob, EM(); dt = 0.01, rng = StableRNG(1), alias = copy_state)
    integrator2 = init(jprob, EM(); dt = 0.01, rng = StableRNG(2), alias = copy_state)
    for _ in 1:50
        step!(integrator1)
    end
    solve!(integrator2)
    solve!(integrator1)
    @test completed(expected1)
    @test completed(expected2)
    @test same_path(integrator1.sol, expected1)
    @test same_path(integrator2.sol, expected2)
end

@testset "Mass-action rates follow parameter changes in jump-diffusions" begin
    birth = MassActionJump([[0 => 1]], [[1 => 1]]; param_idxs = [1])
    jprob = JumpProblem(SDEProblem(f!, no_noise!, [0.0], (0.0, T_END), [0.0]), Direct(),
        birth)
    inactive_sol = solve(jprob, EM(); dt = 0.01, rng = StableRNG(1))
    @test completed(inactive_sol)
    @test inactive_sol.u[end] == [0.0]

    # Remade parameters are used at the next solve.
    active_sol = solve(remake(jprob; p = [50.0]), EM(); dt = 0.01, rng = StableRNG(1))
    @test completed(active_sol)
    @test active_sol.u[end][1] > 0

    # A parameter change in a callback takes effect after `reset_aggregated_jumps!`.
    state_at_stop = Ref([NaN])
    condition(u, t, integrator) = t == 1.0
    function stop_births!(integrator)
        integrator.p[1] = 0.0
        reset_aggregated_jumps!(integrator)
        state_at_stop[] = copy(integrator.u)
        nothing
    end
    sol = solve(remake(jprob; p = [50.0]), EM(); dt = 0.01, rng = StableRNG(1),
        tstops = [1.0], callback = DiscreteCallback(condition, stop_births!))
    @test completed(sol)
    @test state_at_stop[][1] > 0
    @test sol.u[end] == state_at_stop[]
end
