using DiffEqJump, StochasticDiffEq, DiffEqMonteCarlo, Test
# function f(du,u,p,t)
#   du[1] = u[1]
# end
f = (du,u,p,t) -> (du[1]=u[1])
g = (du,u,p,t) -> (du[1]=u[1])
prob = SDEProblem(f,g,[1.0],(0.0,1.0))
rate = (u,p,t) -> 2
affect! = integrator -> (integrator.u[1] = integrator.u[1]/2)
jump = VariableRateJump(rate, affect!, save_positions=(false,true))
jump_prob = JumpProblem(prob,Direct(),jump)
monte_prob = MonteCarloProblem(jump_prob)
sol = solve(monte_prob,SRIW1(),num_monte=3,parallel_type=:none,
            save_everystep=false,dt=0.001,adaptive=false)
@test sol[1].t[2] != sol[2].t[2] != sol[3].t[2]

jump = ConstantRateJump(rate, affect!)
jump_prob = JumpProblem(prob,Direct(),jump,save_positions=(true,false))
monte_prob = MonteCarloProblem(jump_prob)
sol = solve(monte_prob,SRIW1(),num_monte=3,parallel_type=:none,
            save_everystep=false,dt=0.001,adaptive=false)
@test sol[1].t[2] != sol[2].t[2] != sol[3].t[2]
