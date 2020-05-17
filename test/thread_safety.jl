using DiffEqBase
using DiffEqJump
sr = [1.0,2.0,50.]
maj = MassActionJump(sr,[[1 => 1],[1 => 1],[0 => 1]], [[1 => 1], [1 => -1], [1 => 1]])
params = (1.0,2.0,50.)
tspan = (0.,4.)
u0 = [5]
dprob = DiscreteProblem(u0, tspan, params)
jprob = JumpProblem(dprob, Direct(), maj)
solve(EnsembleProblem(jprob), SSAStepper(), EnsembleThreads(); trajectories=10)
solve(EnsembleProblem(jprob;safetycopy=true), SSAStepper(), EnsembleThreads(); trajectories=10)
