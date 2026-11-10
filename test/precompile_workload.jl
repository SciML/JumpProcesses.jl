using JumpProcesses, DiffEqBase
using Random
using SciMLBase: get_rng
using Test

rate = (u, p, t) -> u[1]
affect! = integrator -> (integrator.u[1] += 1)
jump = ConstantRateJump(rate, affect!)
prob = DiscreteProblem([10.0], (0.0, 1.0))
jump_prob = JumpProblem(prob, Direct(), jump; save_positions = (false, false))

@testset "Precompile workload seed=$seed" for seed in (nothing, 12345)
    kwargs = seed === nothing ? (;) : (; seed)
    integrator = init(jump_prob, SSAStepper(); kwargs...)
    if seed === nothing
        @test get_rng(integrator) isa Random.TaskLocalRNG
    else
        @test get_rng(integrator) isa Random.Xoshiro
    end
    step!(integrator)
    @test integrator.t > 0.0
    @test integrator.u[1] >= 10.0

    sol = solve(jump_prob, SSAStepper(); kwargs...)
    @test sol.retcode == ReturnCode.Success
    @test sol.t[end] == 1.0
    @test sol.u[end][1] >= 10.0
end
