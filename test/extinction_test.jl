using DiffEqBase, DiffEqJump, StaticArrays
using Test

reactstoch = [
    [1 => 1]
]

netstoch = [
    [1 => -1]
]

rates = [1.]
spec_to_dep_jumps = [[1]]
jump_to_dep_specs = [[1]]
dg = [[1]]
majump = MassActionJump(rates, reactstoch, netstoch)
u0 = [10]
dprob = DiscreteProblem(u0,(0.,100.),rates)
algs = DiffEqJump.JUMP_AGGREGATORS

for ssa in algs
    local jprob = JumpProblem(dprob, ssa, majump; vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs, save_positions=(false,false))
    local sol = solve(jprob, SSAStepper(), saveat=100.)
    @test sol[1,end] == 0
    @test sol.t[end] < Inf
end

u0 = SA[10]
dprob = DiscreteProblem(u0,(0.,100.),rates)

for ssa in algs
    local jprob = JumpProblem(dprob, ssa, majump; vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs, save_positions=(false,false))
    local sol = solve(jprob, SSAStepper(), saveat=100.)
    @test sol[1,end] == 0
    @test sol.t[end] < Inf
end


# test callback
function extinction_condition(u,t,integrator)   
    integrator.cb.affect!.next_jump_time == Inf
end
function extinction_affect!(integrator)
    (saved, savedexactly) = savevalues!(integrator, true)
    @test saved == true
    @test savedexactly == true
    nothing
end
cb = DiscreteCallback(extinction_condition, extinction_affect!, save_positions=(false,false))
dprob = DiscreteProblem(u0,(0.,1000.),rates)
jprob = JumpProblem(dprob, Direct(), majump; save_positions=(false,false))
sol = solve(jprob, SSAStepper(), callback=cb, save_end=false)
@test sol.t[end] < 1000.0

# test terminate
function extinction_condition(u,t,integrator)   
    u[1] == 1    
end
function extinction_affect!(integrator)
    (saved, savedexactly) = savevalues!(integrator, true)
    terminate!(integrator)
    nothing
end
cb = DiscreteCallback(extinction_condition, extinction_affect!, save_positions=(false,false))
dprob = DiscreteProblem(u0,(0.,1000.),rates)
jprob = JumpProblem(dprob, Direct(), majump; save_positions=(false,false))
sol = solve(jprob, SSAStepper(), callback=cb, save_end=false)
@test sol[1,end] == 1
@test sol.retcode == :Terminated
