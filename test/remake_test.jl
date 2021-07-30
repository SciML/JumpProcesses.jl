using DiffEqJump, DiffEqBase
using StableRNGs
rng = StableRNG(12345)

rate = (u,p,t) -> p[1]*u[1]*u[2]
affect! = function (integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump = ConstantRateJump(rate,affect!)

rate = (u,p,t) -> p[2]*u[2]
affect! = function (integrator)
  integrator.u[2] -= 1
  integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate,affect!)

u0 = [999,1,0]
p = (.1/1000,.01)
tspan = (0.0,2500.0)

dprob  = DiscreteProblem(u0,tspan,p)
jprob = JumpProblem(dprob,Direct(),jump,jump2,save_positions=(false,false), rng=rng)
sol = solve(jprob, SSAStepper())
@test sol[3,end] == 1000

u02 = [1000,1,0]
p2 = (.1/1000,0.0)
dprob2 = remake(dprob, u0=u02, p=p2)
jprob2 = remake(jprob, prob=dprob2)
sol2 = solve(jprob2, SSAStepper())
@test sol2[2,end] == 1001

tspan2 = (0.0, 25000.0)
jprob3 = remake(jprob, p=p2, tspan=tspan2)
sol3 = solve(jprob3, SSAStepper())
@test sol3[2,end] == 1000
@test sol3.t[end] == 25000.0

################ test changing MassActionJump parameters
rng = StableRNG(12345)
rs = [[2=>1],[1 => 1, 2 => 1]]
ns = [[2 => -1, 3 => 1],[1 => -1, 2 => 1]]
pidxs = [2,1]
maj   = MassActionJump(rs, ns; param_idxs=pidxs)
dprob = DiscreteProblem(u0,tspan,p)
jprob = JumpProblem(dprob, Direct(), maj, save_positions=(false,false), rng=rng)
sol   = solve(jprob, SSAStepper())
@test sol[3,end] == 1000

# update the MassActionJump
dprob2 = remake(dprob, u0=u02, p=p2)
jprob2 = remake(jprob, prob=dprob2)
sol2 = solve(jprob2, SSAStepper())
@test sol2[2,end] == 1001

tspan2 = (0.0, 25000.0)
jprob3 = remake(jprob, p=p2, tspan=tspan2)
sol3 = solve(jprob3, SSAStepper())
@test sol3[2,end] == 1000
@test sol3.t[end] == 25000.0

#################

# test error handling
@test_throws ErrorException jprob4 = remake(jprob, prob=dprob2, p=p2)
@test_throws ErrorException jprob5 = remake(jprob, aggregator=RSSA())