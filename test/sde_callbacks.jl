using Random: Xoshiro
using SciMLBase: DiscreteCallback, SDEProblem
using StochasticDiffEq: SOSRI
using JumpProcesses
using Test

@testset "SDE problem callbacks are not duplicated" begin
    f!(du, u, p, t) = (du[1] = 0.0)
    g!(du, u, p, t) = (du[1] = 0.0)
    callback_count = Ref(0)
    condition(u, t, integrator) = t == 0.5
    function affect!(integrator)
        callback_count[] += 1
        integrator.u[1] += 10.0
    end
    callback = DiscreteCallback(condition, affect!)
    problem = SDEProblem(f!, g!, [0.0], (0.0, 1.0))
    jump = ConstantRateJump((u, p, t) -> 0.0, integrator -> nothing)
    jump_problem = JumpProblem(
        problem, Direct(), jump; callback, rng = Xoshiro(12345), tstops = [0.5]
    )

    solution = solve(jump_problem, SOSRI())

    @test callback_count[] == 1
    @test solution.u[end][1] == 10.0
end
