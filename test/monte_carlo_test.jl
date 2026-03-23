using JumpProcesses, StochasticDiffEq, Test
using StableRNGs
rng = StableRNG(12345)

f = (du, u, p, t) -> (du[1] = u[1])
g = (du, u, p, t) -> (du[1] = u[1])
prob = SDEProblem(f, g, [1.0], (0.0, 1.0))
rate = (u, p, t) -> 200.0
affect! = integrator -> (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate, affect!, save_positions = (false, true))
jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VR_FRM())
monte_prob = EnsembleProblem(jump_prob)
sol = solve(monte_prob, SRIW1(), EnsembleSerial(); trajectories = 3,
    save_everystep = false, dt = 0.001, adaptive = false, rng)
first_event(traj) = traj.t[findfirst(>(traj.t[1]), traj.t)]
@test first_event(sol.u[1]) != first_event(sol.u[2]) != first_event(sol.u[3])

jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VR_Direct())
monte_prob = EnsembleProblem(jump_prob)
sol = solve(monte_prob, SRIW1(), EnsembleSerial(); trajectories = 3,
    save_everystep = false, dt = 0.001, adaptive = false, rng)
@test allunique(sol.u[1].t)

jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VR_DirectFW())
monte_prob = EnsembleProblem(jump_prob)
sol = solve(monte_prob, SRIW1(), EnsembleSerial(); trajectories = 3,
    save_everystep = false, dt = 0.001, adaptive = false, rng)
@test allunique(sol.u[1].t)

jump = ConstantRateJump(rate, affect!)
jump_prob = JumpProblem(prob, Direct(), jump; save_positions = (true, false))
monte_prob = EnsembleProblem(jump_prob)
sol = solve(monte_prob, SRIW1(), EnsembleSerial(); trajectories = 3,
    save_everystep = false, dt = 0.001, adaptive = false, rng)
@test sol.u[1].t[2] != sol.u[2].t[2] != sol.u[3].t[2]
