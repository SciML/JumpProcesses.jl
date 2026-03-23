using OrdinaryDiffEq, JumpProcesses, Test
using StableRNGs, Random

j1 = ConstantRateJump((u, p, t) -> 10, (integrator) -> integrator.u[1] += 1)
j2 = ConstantRateJump((u, p, t) -> 1u[1], (integrator) -> integrator.u[1] -= 1)
u0 = [0]

dprob = DiscreteProblem(u0, (0.0, 100.0))

# Test with FunctionMap - pass rng to solve so trajectories get unique sequences
jump_prob = JumpProblem(dprob, Direct(), j1, j2)
ensemble_prob = EnsembleProblem(jump_prob)
sol = solve(ensemble_prob, FunctionMap(), trajectories = 3; rng = StableRNG(12345))
@test Array(sol.u[1]) !== Array(sol.u[2])
@test Array(sol.u[1]) !== Array(sol.u[3])
@test Array(sol.u[2]) !== Array(sol.u[3])
@test eltype(sol.u[1].u[1]) == Int

# Test with SSAStepper - pass rng to solve so trajectories get unique sequences
jump_prob = JumpProblem(dprob, Direct(), j1, j2)
ensemble_prob2 = EnsembleProblem(jump_prob)
sol = solve(ensemble_prob2, SSAStepper(), trajectories = 3; rng = StableRNG(12345))
@test Array(sol.u[1]) !== Array(sol.u[2])
@test Array(sol.u[1]) !== Array(sol.u[3])
@test Array(sol.u[2]) !== Array(sol.u[3])
@test eltype(sol.u[1].u[1]) == Int
