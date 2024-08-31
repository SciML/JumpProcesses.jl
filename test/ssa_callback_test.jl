using JumpProcesses, DiffEqBase, DiffEqCallbacks
using Test
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> u[1]
affect! = function (integrator)
    integrator.u[1] -= 1
    integrator.u[2] += 1
end
jump = ConstantRateJump(rate, affect!)

prob = DiscreteProblem([0.0, 0.0], (0.0, 10.0))
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)

sol = solve(jump_prob, SSAStepper())

@test sol.t == [0.0, 10.0]
@test sol.u == [[0.0, 0.0], [0.0, 0.0]]

condition(u, t, integrator) = t == 5
function fuel_affect!(integrator)
    integrator.u[1] += 100
    reset_aggregated_jumps!(integrator)
end
cb = DiscreteCallback(condition, fuel_affect!, save_positions = (false, true))

sol = solve(jump_prob, SSAStepper(); callback = cb, tstops = [5])
@test sol.t[1:2] == [0.0, 5.0] # no jumps between t=0 and t=5
@test sol(5 + 1e-10) == [100, 0] # state just after fueling before any decays can happen

# test can pass callbacks via JumpProblem
jump_prob2 = JumpProblem(prob, Direct(), jump; rng = rng, callback = cb)
sol2 = solve(jump_prob2, SSAStepper(); tstops = [5])
@test sol2.t[1:2] == [0.0, 5.0] # no jumps between t=0 and t=5
@test sol2(5 + 1e-10) == [100, 0] # state just after fueling before any decays can happen

# test that callback initializer/finalizer is called and add_tstop! works as expected
random_tstops = rand(rng, 100) .* 10 # 100 random Float64 between 0.0 and 10.0

function fuel_init!(cb, u, t, integrator)
    for tstop in random_tstops
        add_tstop!(integrator, tstop)
    end
    @test issorted(integrator.tstops)
end
finalizer_called = 0
fuel_finalize(cb, u, t, integrator) = global finalizer_called += 1

cb2 = DiscreteCallback(condition, fuel_affect!, initialize = fuel_init!,
    finalize = fuel_finalize)
sol = solve(jump_prob, SSAStepper(), callback = cb2)
for tstop in random_tstops
    @test tstop ∈ sol.t
end
@test finalizer_called == 1

# test for updating MassActionJump parameters
rs = [[1 => 1], [2 => 1]]
ns = [[1 => -1, 2 => 1], [1 => 1, 2 => -1]]
p = [1.0, 0.0]
maj = MassActionJump(rs, ns; param_idxs = [1, 2])
u₀ = [100, 0]
tspan = (0.0, 2000.0)
dprob = DiscreteProblem(u₀, tspan, p)
jprob = JumpProblem(dprob, Direct(), maj, save_positions = (false, false), rng = rng)
pcondit(u, t, integrator) = t == 1000.0
function paffect!(integrator)
    integrator.p[1] = 0.0
    integrator.p[2] = 1.0
    reset_aggregated_jumps!(integrator)
end
sol = solve(jprob, SSAStepper(), tstops = [1000.0],
    callback = DiscreteCallback(pcondit, paffect!))
@test all(p .== [0.0, 1.0])
@test sol[1, end] == 100

p .= [1.0, 0.0]
maj1 = MassActionJump([1 => 1], [1 => -1, 2 => 1]; param_idxs = 1)
maj2 = MassActionJump([2 => 1], [1 => 1, 2 => -1]; param_idxs = 2)
jprob = JumpProblem(dprob, Direct(), maj1, maj2, save_positions = (false, false), rng = rng)
sol = solve(jprob, SSAStepper(), tstops = [1000.0],
    callback = DiscreteCallback(pcondit, paffect!))
@test all(p .== [0.0, 1.0])
@test sol[1, end] == 100

p2 = [1.0, 0.0, 0.0]
maj3 = MassActionJump([1 => 1], [1 => -1, 2 => 1]; param_idxs = 3)
dprob = DiscreteProblem(u₀, tspan, p2)
jprob = JumpProblem(dprob, Direct(), maj1, maj2, maj3, save_positions = (false, false),
    rng = rng)
sol = solve(jprob, SSAStepper(), tstops = [1000.0],
    callback = DiscreteCallback(pcondit, paffect!))
@test all(p2 .== [0.0, 1.0, 0.0])
@test sol[1, end] == 100

p2 .= [1.0, 0.0, 0.0]
jprob = JumpProblem(dprob, Direct(), JumpSet(; massaction_jumps = [maj1, maj2, maj3]),
    save_positions = (false, false), rng = rng)
sol = solve(jprob, SSAStepper(), tstops = [1000.0],
    callback = DiscreteCallback(pcondit, paffect!))
@test all(p2 .== [0.0, 1.0, 0.0])
@test sol[1, end] == 100

p .= [1.0, 0.0]
dprob = DiscreteProblem(u₀, tspan, p)
maj4 = MassActionJump([[1 => 1], [2 => 1]], [[1 => -1, 2 => 1], [1 => 1, 2 => -1]];
    param_idxs = [1, 2])
jprob = JumpProblem(dprob, Direct(), maj4, save_positions = (false, false), rng = rng)
sol = solve(jprob, SSAStepper(), tstops = [1000.0],
    callback = DiscreteCallback(pcondit, paffect!))
@test all(p .== [0.0, 1.0])
@test sol[1, end] == 100

# test scale_rates kwarg
p .= [1.0]
dprob = DiscreteProblem(u₀, tspan, p)
maj5 = MassActionJump([[1 => 2]], [[1 => -1, 2 => 1]]; param_idxs = [1])
jprob = JumpProblem(dprob, Direct(), maj5, save_positions = (false, false), rng = rng)
@test all(jprob.massaction_jump.scaled_rates .== [0.5])
jprob = JumpProblem(dprob, Direct(), maj5, save_positions = (false, false), rng = rng,
    scale_rates = false)
@test all(jprob.massaction_jump.scaled_rates .== [1.0])

# test for https://github.com/SciML/JumpProcesses.jl/issues/239
maj6 = MassActionJump([[1 => 1], [2 => 1]], [[1 => -1, 2 => 1], [1 => 1, 2 => -1]];
    param_idxs = [1, 2])
p = (0.1, 0.1)
dprob = DiscreteProblem([10, 0], (0.0, 100.0), p)
jprob = JumpProblem(dprob, Direct(), maj6; save_positions = (false, false), rng = rng)
cbtimes = [20.0, 30.0]
affectpresets!(integrator) = integrator.u[1] += 10
cb = PresetTimeCallback(cbtimes, affectpresets!)
jsol = solve(jprob, SSAStepper(), saveat = 0.1, callback = cb)
@test (jsol(20.00000000001) - jsol(19.9999999999))[1] == 10

# test periodic callbacks working, i.e. #417
let
    rate(u, p, t) = 0.0
    affect!(integ) = (nothing)
    crj = ConstantRateJump(rate, affect!)
    dprob = DiscreteProblem([0], (0.0, 10.0))
    cbfun(integ) = (integ.u[1] += 1; nothing)
    cb = PeriodicCallback(cbfun, 1.0)
    jprob = JumpProblem(dprob, crj; rng)
    sol = solve(jprob; callback = cb)
    @test sol[1, end] == 9

    cb = PeriodicCallback(cbfun, 1.0; initial_affect = true)
    jprob = JumpProblem(dprob, crj; rng)
    sol = solve(jprob; callback = cb)
    @test sol[1, end] == 10

    cb = PeriodicCallback(cbfun, 1.0; initial_affect = true, final_affect = true)
    jprob = JumpProblem(dprob, crj; rng)
    sol = solve(jprob; callback = cb)
    @test sol[1, end] == 11
end

# test for tstops aliasing, i.e.#442
let
    rate(u, p, t) = 0.0
    affect!(integ) = (nothing)
    crj = ConstantRateJump(rate, affect!)
    dprob = DiscreteProblem([0], (0.0, 10.0))
    cbfun(integ) = (integ.u[1] += 1; nothing)
    cb = PeriodicCallback(cbfun, 1.0)
    jprob = JumpProblem(dprob, crj; rng)
    tstops = Float64[]
    # tests for when aliasing system is in place
    #sol = solve(jprob; callback = cb, tstops, alias_tstops = true) 
    # @test sol[1, end] == 9
    #@test tstops == 1.0:9.0    
    # empty!(tstops)
    # sol = solve(jprob; callback = cb, tstops, alias_tstops = false)
    # @test sol[1, end] == 9
    # @test isempty(tstops)
    sol = solve(jprob; callback = cb, tstops)
    @test sol[1, end] == 9
    @test isempty(tstops)

    empty!(tstops)
    integ = init(jprob, SSAStepper(); callback = cb, tstops)
    solve!(integ)
    @test integ.tstops !== tstops
    @test isempty(tstops)
end
