using JumpProcesses, StochasticDiffEq, Test
using StableRNGs
rng = StableRNG(12345)

f = (du, u, p, t) -> (du[1] = u[1])
g = (du, u, p, t) -> (du[1] = u[1])
prob = SDEProblem(f, g, [1.0], (0.0, 1.0))
rate = (u, p, t) -> 200.0
affect! = integrator -> (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate, affect!, save_positions = (false, true))
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
monte_prob = EnsembleProblem(jump_prob)
sol = solve(monte_prob, SRIW1(), EnsembleSerial(), trajectories = 3,
    save_everystep = false, dt = 0.001, adaptive = false)
@test sol.u[1].t[2] != sol.u[2].t[2]

jump = ConstantRateJump(rate, affect!)
jump_prob = JumpProblem(prob, Direct(), jump, save_positions = (true, false), rng = rng)
monte_prob = EnsembleProblem(jump_prob)
sol = solve(monte_prob, SRIW1(), EnsembleSerial(), trajectories = 3,
    save_everystep = false, dt = 0.001, adaptive = false)
@test sol.u[1].t[2] != sol.u[2].t[2] != sol.u[3].t[2]
