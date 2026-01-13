using OrdinaryDiffEq, JumpProcesses, Test
using StableRNGs, Random

j1 = ConstantRateJump((u, p, t) -> 10, (integrator) -> integrator.u[1] += 1)
j2 = ConstantRateJump((u, p, t) -> 1u[1], (integrator) -> integrator.u[1] -= 1)
u0 = [0]

dprob = DiscreteProblem(u0, (0.0, 100.0))

# For EnsembleProblems, use prob_func to create a new JumpProblem with unique RNG per trajectory.
# This ensures different trajectories while maintaining reproducibility.
# Generate seeds from a seeded RNG for reproducibility of ensemble results.
function make_seeded_prob_func(dprob, aggregator, jumps, base_rng)
    return function prob_func(prob, i, repeat)
        seed = rand(base_rng, UInt64)
        JumpProblem(dprob, aggregator, jumps...; rng = StableRNG(seed))
    end
end

# Test with FunctionMap - use prob_func to create JumpProblems with unique RNGs
rng1 = StableRNG(12345)
jump_prob = JumpProblem(dprob, Direct(), j1, j2; rng = rng1)
ensemble_rng = StableRNG(99999)  # separate RNG for generating trajectory seeds
ensemble_prob = EnsembleProblem(jump_prob;
    prob_func = make_seeded_prob_func(dprob, Direct(), (j1, j2), ensemble_rng))
sol = solve(ensemble_prob, FunctionMap(), trajectories = 3)
@test Array(sol.u[1]) !== Array(sol.u[2])
@test Array(sol.u[1]) !== Array(sol.u[3])
@test Array(sol.u[2]) !== Array(sol.u[3])
@test eltype(sol.u[1].u[1]) == Int

# Test with SSAStepper - use prob_func to create JumpProblems with unique RNGs
rng2 = StableRNG(12345)
jump_prob = JumpProblem(dprob, Direct(), j1, j2; rng = rng2)
ensemble_rng2 = StableRNG(99999)  # separate RNG for generating trajectory seeds
ensemble_prob2 = EnsembleProblem(jump_prob;
    prob_func = make_seeded_prob_func(dprob, Direct(), (j1, j2), ensemble_rng2))
sol = solve(ensemble_prob2, SSAStepper(), trajectories = 3)
@test Array(sol.u[1]) !== Array(sol.u[2])
@test Array(sol.u[1]) !== Array(sol.u[3])
@test Array(sol.u[2]) !== Array(sol.u[3])
@test eltype(sol.u[1].u[1]) == Int
