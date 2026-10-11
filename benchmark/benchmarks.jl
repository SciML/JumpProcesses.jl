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

# The benchmark job runs this script against both this branch and the base branch.
# JumpProcesses 10 takes the RNG at `solve`, and `JumpProblem(...; rng)` raises an
# `ArgumentError`; earlier versions store it in the `JumpProblem` and reject it in `solve`.
# Either way, every sample simulates the same trajectory, from a `StableRNG` seeded with
# 12345, so the two branches' timings are comparable.
const RNG_AT_SOLVE = try
    JumpProblem(dprob, Direct(), maj; rng = StableRNG(1))
    false
catch err
    err isa ArgumentError || rethrow()
    true
end

if RNG_AT_SOLVE
    jump_problem(args...) = JumpProblem(args...)
    function benchmark_solve(jprob)
        @benchmarkable solve($jprob, SSAStepper(); rng) setup=(rng=StableRNG(12345)) evals=1
    end
else
    # `seed` reseeds the problem's RNG.
    jump_problem(args...) = JumpProblem(args...; rng = StableRNG(12345))
    function benchmark_solve(jprob)
        @benchmarkable solve($jprob, SSAStepper(); seed = 12345) evals=1
    end
end

# =============================================================================
# Solves (SSAStepper, seeded RNG)
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

jprob_maj = jump_problem(dprob, Direct(), maj)
jprob_cr = jump_problem(dprob, Direct(), jump1, jump2)

SUITE["solve"]["massaction"] = benchmark_solve(jprob_maj)
SUITE["solve"]["constantrate"] = benchmark_solve(jprob_cr)

# =============================================================================
# Aggregators
# =============================================================================

SUITE["aggregators"] = BenchmarkGroup()

jprob_rdirect = jump_problem(dprob, RDirect(), maj)
jprob_sorting = jump_problem(dprob, SortingDirect(), maj)
jprob_nrm = jump_problem(dprob, NRM(), maj)

SUITE["aggregators"]["RDirect"] = benchmark_solve(jprob_rdirect)
SUITE["aggregators"]["SortingDirect"] = benchmark_solve(jprob_sorting)
SUITE["aggregators"]["NRM"] = benchmark_solve(jprob_nrm)
