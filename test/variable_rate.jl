using DiffEqBase, DiffEqJump, OrdinaryDiffEq

rate = (t,u) -> u[1]
affect! = (integrator) -> (integrator.u[1] = integrator.u[1]/2)

jump = VariableRateJump(rate,affect!)

jumps = JumpSet(jump)

condition = function (t,u,integrator)
  u.jump_u[1]
end

callback_affect! = function (integrator)
  affect!(integrator)
  integrator.u.jump_u[1] = -randexp()
end

cb = ContinuousCallback(condition,callback_affect!)

f = function (t,u,du)
  du[1] = u[1]
end

u0 = ExtendedJumpArray([0.2],[-randexp()])


prob = ODEProblem(f,u0,(0.0,10.0))
jump_prob = JumpProblem(prob,Direct(),jump)

sol = solve(jump_prob,Tsit5(),callback=cb)
