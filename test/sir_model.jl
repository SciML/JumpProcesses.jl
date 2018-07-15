using DiffEqJump, DiffEqBase, OrdinaryDiffEq
using Test

rate = (u,p,t) -> (0.1/1000.0)*u[1]*u[2]
affect! = function (integrator)
  integrator.u[1] -= 1
  integrator.u[2] += 1
end
jump = ConstantRateJump(rate,affect!)

rate = (u,p,t) -> 0.01u[2]
affect! = function (integrator)
  integrator.u[2] -= 1
  integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate,affect!)


prob = DiscreteProblem([999.0,1.0,0.0],(0.0,250.0))
jump_prob = JumpProblem(prob,Direct(),jump,jump2)
integrator = init(jump_prob,FunctionMap())
sol = solve(jump_prob,FunctionMap())

jump_prob

using Plots; plotly(); plot(sol)

nums = Int[]
@time for i in 1:1000
  sol = solve(jump_prob,FunctionMap())
  push!(nums,sol[end][1])
end
mean(nums)

using ProfileView
@profile for i in 1:1000; solve(jump_prob,FunctionMap()); end
Profile.clear()
@profile for i in 1:1000; solve(jump_prob,FunctionMap()); end
ProfileView.view()
