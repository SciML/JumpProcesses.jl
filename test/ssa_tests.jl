using DiffEqJump, DiffEqBase
using Test

rate = (u,p,t) -> u[1]
affect! = function (integrator)
  integrator.u[1] += 1
end
jump = ConstantRateJump(rate,affect!)

rate = (u,p,t) -> 0.5u[1]
affect! = function (integrator)
  integrator.u[1] -= 1
end
jump2 = ConstantRateJump(rate,affect!)

prob = DiscreteProblem([10.0],(0.0,3.0))
jump_prob = JumpProblem(prob,Direct(),jump,jump2)

integrator = init(jump_prob,SSAStepper())
step!(integrator)
integrator.u[1]

sol = solve(jump_prob,SSAStepper())

jump_prob = JumpProblem(prob,Direct(),jump,jump2,save_positions=(false,false))
sol = solve(jump_prob,SSAStepper(),saveat=0.0:0.1:2.9)
@test sol.t == collect(0.0:0.1:3.0)
