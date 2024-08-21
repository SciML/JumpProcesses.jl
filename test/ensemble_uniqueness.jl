using OrdinaryDiffEq, JumpProcesses, Test
using StableRNGs
rng = StableRNG(12345)

j1 = ConstantRateJump((u, p, t) -> 10, (integrator) -> integrator.u[1] += 1)
j2 = ConstantRateJump((u, p, t) -> 1u[1], (integrator) -> integrator.u[1] -= 1)
u0 = [0]

prob = DiscreteProblem(u0, (0.0, 100.0))
jump_prob = JumpProblem(prob, Direct(), j1, j2; rng = rng)
sol = solve(EnsembleProblem(jump_prob), FunctionMap(), trajectories = 3)
@test Array(sol.u[1]) !== Array(sol.u[2])
@test Array(sol.u[1]) !== Array(sol.u[3])
@test Array(sol.u[2]) !== Array(sol.u[3])
@test eltype(sol.u[1].u[1]) == Int
sol = solve(EnsembleProblem(jump_prob), SSAStepper(), trajectories = 3)
@test Array(sol.u[1]) !== Array(sol.u[2])
@test Array(sol.u[1]) !== Array(sol.u[3])
@test Array(sol.u[2]) !== Array(sol.u[3])
@test eltype(sol.u[1].u[1]) == Int
