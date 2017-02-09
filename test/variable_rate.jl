using DiffEqBase, DiffEqJump, OrdinaryDiffEq

rate = (t,u) -> u[1]
affect! = (integrator) -> (integrator.u[1] = integrator.u[1]/2)

jump = VariableRateJump(rate,affect!)
jump2 = deepcopy(jump)

f = function (t,u,du)
  du[1] = u[1]
end

prob = ODEProblem(f,[0.2],(0.0,10.0))
jump_prob = JumpProblem(prob,Direct(),jump,jump2)

sol = solve(jump_prob,Tsit5())
