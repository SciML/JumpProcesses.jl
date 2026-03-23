using JumpProcesses, StaticArrays
using Test
using StableRNGs
rng = StableRNG(12345)

reactstoch = [
    [1 => 1]
]

netstoch = [
    [1 => -1]
]

Nsims = 10
rates = [1.0]
dg = [[1]]
majump = MassActionJump(rates, reactstoch, netstoch)
u0 = [100000]
dprob = DiscreteProblem(u0, (0.0, 1e5), rates)
algs = (JumpProcesses.JUMP_AGGREGATORS..., JumpProcesses.NullAggregator())

for n in 1:Nsims
    for ssa in algs
        local jprob = JumpProblem(dprob, ssa, majump, save_positions = (false, false))
        local sol = solve(jprob, SSAStepper(); rng)
        @test sol[1, end] == 0
        @test sol.t[end] < Inf
    end
end

u0 = SA[10]
dprob = DiscreteProblem(u0, (0.0, 100.0), rates)

for ssa in algs
    local jprob = JumpProblem(dprob, ssa, majump, save_positions = (false, false))
    local sol = solve(jprob, SSAStepper(); saveat = 100.0, rng)
    @test sol[1, end] == 0
    @test sol.t[end] < Inf
end

# test callback
Base.@kwdef mutable struct ExtinctionTest
    cnt::Int = 0
end
function (e::ExtinctionTest)(u, t, integrator)
    (e.cnt == 0) && (integrator.cb.affect!.next_jump_time == Inf)
end
function (e::ExtinctionTest)(integrator)
    (saved, savedexactly) = savevalues!(integrator, true)
    @test saved == true
    @test savedexactly == true
    e.cnt += 1
    nothing
end
et = ExtinctionTest()
cb = DiscreteCallback(et, et, save_positions = (false, false))
dprob = DiscreteProblem(u0, (0.0, 1000.0), rates)
jprob = JumpProblem(dprob, Direct(), majump; save_positions = (false, false))
sol = solve(jprob, SSAStepper(); callback = cb, save_end = false, rng)
@test sol.t[end] < 1000.0

# test terminate
function extinction_condition2(u, t, integrator)
    u[1] == 1
end
function extinction_affect!2(integrator)
    (saved, savedexactly) = savevalues!(integrator, true)
    terminate!(integrator)
    nothing
end
cb = DiscreteCallback(extinction_condition2, extinction_affect!2,
    save_positions = (false, false))
dprob = DiscreteProblem(u0, (0.0, 1000.0), rates)
jprob = JumpProblem(dprob, majump; save_positions = (false, false))
sol = solve(jprob; callback = cb, save_end = false, rng)
@test sol[1, end] == 1
@test sol.retcode == ReturnCode.Terminated
@test sol.t[end] < 1000.0
