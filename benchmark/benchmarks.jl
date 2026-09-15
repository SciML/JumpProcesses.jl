using JumpProcesses, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()

# SIR model: S + I -> 2I at rate p[1], I -> R at rate p[2]
p = (1.0e-4, 0.01)
u0 = [999, 1, 0]
tspan = (0.0, 250.0)
dprob = DiscreteProblem(u0, tspan, p)

pidxs = [1, 2]
substoich = [[1 => 1, 2 => 1], [2 => 1]]
netstoich = [[1 => -1, 2 => 1], [2 => -1, 3 => 1]]
maj = MassActionJump(substoich, netstoich; param_idxs = pidxs)

rate1(u, p, t) = p[1] * u[1] * u[2]
function affect1!(integrator)
    integrator.u[1] -= 1
    return integrator.u[2] += 1
end
jump1 = ConstantRateJump(rate1, affect1!)

rate2(u, p, t) = p[2] * u[2]
function affect2!(integrator)
    integrator.u[2] -= 1
    return integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate2, affect2!)

# =============================================================================
# Problem construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["jumpproblem_massaction"] = @benchmarkable JumpProblem(
    $dprob, Direct(), $maj
)
SUITE["construct"]["jumpproblem_constantrate"] = @benchmarkable JumpProblem(
    $dprob, Direct(), $jump1, $jump2
)

# =============================================================================
# Solves (SSAStepper, fixed rng for reproducibility)
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

jprob_maj = JumpProblem(dprob, Direct(), maj; rng = StableRNG(12345))
jprob_cr = JumpProblem(dprob, Direct(), jump1, jump2; rng = StableRNG(12345))

SUITE["solve"]["massaction"] = @benchmarkable solve($jprob_maj, SSAStepper())
SUITE["solve"]["constantrate"] = @benchmarkable solve($jprob_cr, SSAStepper())

# =============================================================================
# Aggregators
# =============================================================================

SUITE["aggregators"] = BenchmarkGroup()

jprob_rdirect = JumpProblem(dprob, RDirect(), maj; rng = StableRNG(12345))
jprob_sorting = JumpProblem(dprob, SortingDirect(), maj; rng = StableRNG(12345))
jprob_nrm = JumpProblem(dprob, NRM(), maj; rng = StableRNG(12345))

SUITE["aggregators"]["RDirect"] = @benchmarkable solve($jprob_rdirect, SSAStepper())
SUITE["aggregators"]["SortingDirect"] = @benchmarkable solve(
    $jprob_sorting, SSAStepper()
)
SUITE["aggregators"]["NRM"] = @benchmarkable solve($jprob_nrm, SSAStepper())
