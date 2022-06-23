using DiffEqBase
using DiffEqJump
using StableRNGs
rng = StableRNG(12345)

sr = [1.0, 2.0, 50.0]
maj = MassActionJump(sr, [[1 => 1], [1 => 1], [0 => 1]], [[1 => 1], [1 => -1], [1 => 1]])
params = (1.0, 2.0, 50.0)
tspan = (0.0, 4.0)
u0 = [5]
dprob = DiscreteProblem(u0, tspan, params)
jprob = JumpProblem(dprob, Direct(), maj; rng = rng)
solve(EnsembleProblem(jprob), SSAStepper(), EnsembleThreads(); trajectories = 10)
solve(EnsembleProblem(jprob; safetycopy = true), SSAStepper(), EnsembleThreads();
      trajectories = 10)
