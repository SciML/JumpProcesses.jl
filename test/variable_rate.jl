using DiffEqBase, DiffEqJump, OrdinaryDiffEq

rate = (t,u) -> u[1]
affect! = (integrator) -> (integrator.u[1] = integrator.u[1]/2)

VariableRateJump(rate,affect!)

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

jump_f = function (t,u,du)
  f(t,u,du)
  du[2] = rate(t,u)
end

u0 = ExtendedJumpArray([0.2],[-randexp()])

prob = ODEProblem(jump_f,u0,(0.0,10.0))
sol = solve(prob,Tsit5(),callback=cb)
