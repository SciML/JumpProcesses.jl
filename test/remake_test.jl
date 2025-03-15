using JumpProcesses, DiffEqBase, OrdinaryDiffEq
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> p[1] * u[1] * u[2]
affect! = function (integrator)
    integrator.u[1] -= 1
    integrator.u[2] += 1
end
jump = ConstantRateJump(rate, affect!)

rate = (u, p, t) -> p[2] * u[2]
affect! = function (integrator)
    integrator.u[2] -= 1
    integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate, affect!)

u0 = [999, 1, 0]
p = (0.1 / 1000, 0.01)
tspan = (0.0, 2500.0)

dprob = DiscreteProblem(u0, tspan, p)
jprob = JumpProblem(dprob, Direct(), jump, jump2, save_positions = (false, false),
    rng = rng)
sol = solve(jprob, SSAStepper())
@test sol[3, end] == 1000

u02 = [1000, 1, 0]
p2 = (0.1 / 1000, 0.0)
dprob2 = remake(dprob, u0 = u02, p = p2)
jprob2 = remake(jprob, prob = dprob2)
sol2 = solve(jprob2, SSAStepper())
@test sol2[2, end] == 1001

tspan2 = (0.0, 25000.0)
jprob3 = remake(jprob, p = p2, tspan = tspan2)
sol3 = solve(jprob3, SSAStepper())
@test sol3[2, end] == 1000
@test sol3.t[end] == 25000.0

################ test changing MassActionJump parameters
rng = StableRNG(12345)
rs = [[2 => 1], [1 => 1, 2 => 1]]
ns = [[2 => -1, 3 => 1], [1 => -1, 2 => 1]]
pidxs = [2, 1]
maj = MassActionJump(rs, ns; param_idxs = pidxs)
dprob = DiscreteProblem(u0, tspan, p)
jprob = JumpProblem(dprob, Direct(), maj, save_positions = (false, false), rng = rng)
sol = solve(jprob, SSAStepper())
@test sol[3, end] == 1000

# update the MassActionJump
dprob2 = remake(dprob, u0 = u02, p = p2)
jprob2 = remake(jprob, prob = dprob2)
sol2 = solve(jprob2, SSAStepper())
@test sol2[2, end] == 1001

tspan2 = (0.0, 25000.0)
jprob3 = remake(jprob, p = p2, tspan = tspan2)
sol3 = solve(jprob3, SSAStepper())
@test sol3[2, end] == 1000
@test sol3.t[end] == 25000.0

#################

# test error handling
@test_throws ErrorException jprob4=remake(jprob, prob = dprob2, p = p2)
@test_throws ErrorException jprob5=remake(jprob, aggregator = RSSA())

# test for #446
let
    f(du, u, p, t) = (du .= 0; nothing)
    prob = ODEProblem(f, [0.0], (0.0, 1.0))
    rrate(u, p, t) = u[1]
    aaffect!(integrator) = (integrator.u[1] += 1; nothing)
    vrj = VariableRateJump(rrate, aaffect!)
    jprob = JumpProblem(prob, vrj; variablerate_aggregator = NextReactionODE(), rng)
    sol = solve(jprob, Tsit5())
    @test all(==(0.0), sol[1, :])
    u0 = [4.0]
    jprob2 = remake(jprob; u0)
    @test jprob2.prob.u0 isa ExtendedJumpArray
    @test jprob2.prob.u0.u === u0
    sol = solve(jprob2, Tsit5())
    u = sol[1, :]
    @test length(u) > 2
    @test all(>(u0[1]), u[3:end])
    u0 = deepcopy(jprob2.prob.u0)
    u0.u .= 0
    jprob3 = remake(jprob2; u0)
    sol = solve(jprob3, Tsit5())
    @test all(==(0.0), sol[1, :])
    @test_throws ErrorException jprob4=remake(jprob, u0 = 1)
end

# tests when changing u0 via a passed in prob
let
    f(du, u, p, t) = (du .= 0; nothing)
    prob = ODEProblem(f, [0.0], (0.0, 1.0))
    rrate(u, p, t) = u[1]
    aaffect!(integrator) = (integrator.u[1] += 1; nothing)
    vrj = VariableRateJump(rrate, aaffect!)
    jprob = JumpProblem(prob, vrj; variablerate_aggregator = NextReactionODE(), rng)
    sol = solve(jprob, Tsit5())
    @test all(==(0.0), sol[1, :])
    u0 = [4.0]
    prob2 = remake(jprob.prob; u0)
    @test_throws ErrorException jprob2=remake(jprob; prob = prob2)
    u0eja = JumpProcesses.remake_extended_u0(jprob.prob, u0, rng)
    prob3 = remake(jprob.prob; u0 = u0eja)
    jprob3 = remake(jprob; prob = prob3)
    @test jprob3.prob.u0 isa ExtendedJumpArray
    @test jprob3.prob.u0 === u0eja
    sol = solve(jprob3, Tsit5())
    u = sol[1, :]
    @test length(u) > 2
    @test all(>(u0[1]), u[3:end])
end
