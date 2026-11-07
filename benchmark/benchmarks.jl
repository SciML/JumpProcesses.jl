using JumpProcesses, BenchmarkTools
using StableRNGs
using OrdinaryDiffEqTsit5, StochasticDiffEqHighOrder

# Every ODE/SDE problem is FullSpecialize: on Julia >= 1.12 the default specialization makes
# DiffEqBase erase callback types, which allocates on every step, and the suite would time that
# instead of JumpProcesses
const FullSpecialize = JumpProcesses.SciMLBase.FullSpecialize

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
    jump_problem(args...; kwargs...) = JumpProblem(args...; kwargs...)
    function benchmark_solve(jprob, alg = SSAStepper(); seed = 12345, kwargs...)
        @benchmarkable solve($jprob, $alg; rng, $kwargs...) setup=(rng=StableRNG($seed)) evals=1
    end
else
    # `seed` reseeds the problem's RNG.
    function jump_problem(args...; kwargs...)
        JumpProblem(args...; rng = StableRNG(12345), kwargs...)
    end
    function benchmark_solve(jprob, alg = SSAStepper(); seed = 12345, kwargs...)
        @benchmarkable solve($jprob, $alg; seed = $seed, $kwargs...) evals=1
    end
end

aggname(agg) = string(nameof(typeof(agg)))

# =============================================================================
# Solves (SSAStepper, seeded RNG)
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

jprob_maj = jump_problem(dprob, Direct(), maj)
jprob_cr = jump_problem(dprob, Direct(), jump1, jump2)

SUITE["solve"]["massaction"] = benchmark_solve(jprob_maj)
SUITE["solve"]["constantrate"] = benchmark_solve(jprob_cr)

# =============================================================================
# Aggregators: the SIR model with each aggregator, SSAStepper
# =============================================================================

SUITE["aggregators"] = BenchmarkGroup()

for agg in (RDirect(), SortingDirect(), NRM(), DirectFW(), DirectCR(), RSSA(), RSSACR(),
    FRM(), FRMFW(), CCNRM(), Coevolve())
    # with seed 12345, RSSACR's single infected recovers before infecting anyone
    seed = agg isa RSSACR ? 1 : 12345
    SUITE["aggregators"][aggname(agg)] = benchmark_solve(jump_problem(dprob, agg, maj); seed)
end

# =============================================================================
# Pure jump problems, SSAStepper
# =============================================================================

SUITE["pure_jump"] = BenchmarkGroup()

# A linear chain A₁ ⇄ A₂ ⇄ … ⇄ A_N of conversions at rate 1: a large, sparse network, where the
# aggregators that use dependency graphs matter
const NCHAIN = 200
links = 1:(NCHAIN - 1)
chain_substoich = vcat([[i => 1] for i in links], [[i + 1 => 1] for i in links])
chain_netstoich = vcat([[i => -1, i + 1 => 1] for i in links],
    [[i => 1, i + 1 => -1] for i in links])
chain_maj = MassActionJump(ones(2 * (NCHAIN - 1)), chain_substoich, chain_netstoich)
chain_prob = DiscreteProblem(fill(5, NCHAIN), (0.0, 2.0))

SUITE["pure_jump"]["chain"] = BenchmarkGroup()
for agg in (Direct(), SortingDirect(), DirectCR(), RSSA(), RSSACR(), NRM(), CCNRM())
    SUITE["pure_jump"]["chain"][aggname(agg)] = benchmark_solve(
        jump_problem(chain_prob, agg, chain_maj; save_positions = (false, false)))
end

# Bounded VariableRateJumps: 16 jumps with rates 10 (1 + sin(t + i)), bounded by 20, each
# incrementing its own counter
const NBOUNDED = 16
bounded_jumps = [VariableRateJump((u, p, t) -> 10 * (1 + sin(t + i)),
                     integrator -> (integrator.u[i] += 1; nothing);
                     urate = (u, p, t) -> 20.0, rateinterval = (u, p, t) -> typemax(t))
                 for i in 1:NBOUNDED]
bounded_prob = DiscreteProblem(zeros(Int, NBOUNDED), (0.0, 10.0))
SUITE["pure_jump"]["bounded_vrj"] = BenchmarkGroup()
bounded_jprob = jump_problem(bounded_prob, Coevolve(), bounded_jumps...;
    dep_graph = [[i] for i in 1:NBOUNDED], save_positions = (false, false))
SUITE["pure_jump"]["bounded_vrj"]["Coevolve"] = benchmark_solve(bounded_jprob)

# =============================================================================
# General VariableRateJumps in an ODE problem
# =============================================================================

# The VariableRateJump benchmark of #691: N counters and a decaying c, c' = -c/10. Jump i has
# rate λ c (1 + sin(t + i)) and increments counter i; λ = 32 / N gives about 200 jumps.
function m1_problem(N)
    f!(du, u, p, t) = (du .= 0; du[end] = -0.1 * u[end]; nothing)
    ODEProblem{true, FullSpecialize}(f!, [zeros(N); 1.0], (0.0, 10.0), 32.0 / N)
end
function m1_jumps(N; bounded = false)
    map(1:N) do i
        rate(u, p, t) = p * u[end] * (1 + sin(t + i))
        affect!(integrator) = (integrator.u[i] += 1; nothing)
        if bounded
            # c only decays, so 2 λ c(t) bounds the rate from t on
            VariableRateJump(rate, affect!; urate = (u, p, t) -> 2 * p * u[end],
                rateinterval = (u, p, t) -> typemax(t), save_positions = (false, false))
        else
            VariableRateJump(rate, affect!; save_positions = (false, false))
        end
    end
end

SUITE["vrj"] = BenchmarkGroup()
for N in (1, 16)
    group = SUITE["vrj"]["m1_$N"] = BenchmarkGroup()
    for vr_aggregator in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jprob = jump_problem(m1_problem(N), JumpSet(; variable_jumps = m1_jumps(N));
            vr_aggregator)
        group[aggname(vr_aggregator)] = benchmark_solve(jprob, Tsit5();
            save_everystep = false)
    end
end
SUITE["vrj"]["m1_16"]["Coevolve"] = benchmark_solve(
    jump_problem(m1_problem(16), Coevolve(), m1_jumps(16; bounded = true)...;
        dep_graph = [[i] for i in 1:16], save_positions = (false, false)),
    Tsit5(); save_everystep = false)

SUITE["construct"]["vr_frm"] = @benchmarkable JumpProblem($(m1_problem(16)),
    JumpSet(; variable_jumps = $(m1_jumps(16))); vr_aggregator = VR_FRM())
SUITE["construct"]["vr_direct"] = @benchmarkable JumpProblem($(m1_problem(16)),
    JumpSet(; variable_jumps = $(m1_jumps(16))); vr_aggregator = VR_Direct())

# =============================================================================
# Jumps coupled to ODEs and SDEs
# =============================================================================

# Telegraph gene: the promoter switches off ⇄ on (u[1], u[2]) by mass-action jumps at rate 1,
# and the protein u[3] follows p' = 10 on - p / 10, with additive noise in the SDE version
telegraph_maj = MassActionJump([1.0, 1.0], [[1 => 1], [2 => 1]],
    [[1 => -1, 2 => 1], [1 => 1, 2 => -1]])
function telegraph_f!(du, u, p, t)
    du[1] = 0
    du[2] = 0
    du[3] = 10 * u[2] - 0.1 * u[3]
    nothing
end
telegraph_g!(du, u, p, t) = (du[1] = 0; du[2] = 0; du[3] = 1; nothing)
telegraph_ode = ODEProblem{true, FullSpecialize}(
    telegraph_f!, [1.0, 0.0, 0.0], (0.0, 500.0))
telegraph_sde = SDEProblem{true, FullSpecialize}(
    telegraph_f!, telegraph_g!, [1.0, 0.0, 0.0], (0.0, 100.0))
# A VariableRateJump whose rate reads the protein: export of one protein at rate p / 20, while
# there is at least one (the SDE's noise can take p below zero)
export_jump = VariableRateJump((u, p, t) -> u[3] >= 1 ? 0.05 * u[3] : 0.0,
    integrator -> (integrator.u[3] -= 1; nothing); save_positions = (false, false))

SUITE["ode"] = BenchmarkGroup()
SUITE["ode"]["telegraph"] = BenchmarkGroup()
for agg in (Direct(), NRM(), Coevolve())
    SUITE["ode"]["telegraph"][aggname(agg)] = benchmark_solve(
        jump_problem(telegraph_ode, agg, telegraph_maj; save_positions = (false, false)),
        Tsit5(); save_everystep = false)
end
SUITE["ode"]["mixed"] = BenchmarkGroup()
for vr_aggregator in (VR_FRM(), VR_Direct())
    SUITE["ode"]["mixed"][aggname(vr_aggregator)] = benchmark_solve(
        jump_problem(telegraph_ode, Direct(), telegraph_maj, export_jump; vr_aggregator,
            save_positions = (false, false)),
        Tsit5(); save_everystep = false)
end
# The SIR model as an ODE problem with no other dynamics
sir_ode = ODEProblem{true, FullSpecialize}((du, u, p, t) -> (du .= 0; nothing),
    [999.0, 1.0, 0.0], tspan, p)
SUITE["ode"]["sir_zero_drift"] = BenchmarkGroup()
for agg in (Direct(), NRM())
    SUITE["ode"]["sir_zero_drift"][aggname(agg)] = benchmark_solve(
        jump_problem(sir_ode, agg, maj; save_positions = (false, false)),
        Tsit5(); save_everystep = false)
end

SUITE["sde"] = BenchmarkGroup()
SUITE["sde"]["telegraph"] = BenchmarkGroup()
SUITE["sde"]["telegraph"]["Direct"] = benchmark_solve(
    jump_problem(telegraph_sde, Direct(), telegraph_maj; save_positions = (false, false)),
    SRIW1(); save_everystep = false)
SUITE["sde"]["mixed"] = BenchmarkGroup()
for vr_aggregator in (VR_FRM(), VR_Direct())
    SUITE["sde"]["mixed"][aggname(vr_aggregator)] = benchmark_solve(
        jump_problem(telegraph_sde, Direct(), telegraph_maj, export_jump; vr_aggregator,
            save_positions = (false, false)),
        SRIW1(); save_everystep = false)
end

# =============================================================================
# Events
# =============================================================================

SUITE["events"] = BenchmarkGroup()

# The SIR model with 100 susceptibles added at fixed times by a DiscreteCallback, SSAStepper
const DOSING_TIMES = (50.0, 100.0, 150.0, 200.0)
function add_susceptibles!(integrator)
    integrator.u[1] += 100
    reset_aggregated_jumps!(integrator)
    nothing
end
sir_dosing = DiscreteCallback((u, t, integrator) -> t in DOSING_TIMES, add_susceptibles!;
    save_positions = (false, false))
SUITE["events"]["ssa_dosing"] = BenchmarkGroup()
SUITE["events"]["ssa_dosing"]["Direct"] = benchmark_solve(
    jump_problem(dprob, Direct(), maj; save_positions = (false, false)), SSAStepper();
    callback = sir_dosing, tstops = collect(DOSING_TIMES))

# The telegraph gene with a growing protein, p' = 10 on, which a ContinuousCallback halves at
# 100 (cell division)
function telegraph_growth_f!(du, u, p, t)
    du[1] = 0
    du[2] = 0
    du[3] = 10 * u[2]
    nothing
end
division = ContinuousCallback((u, t, integrator) -> u[3] - 100,
    integrator -> (integrator.u[3] /= 2; nothing); save_positions = (false, false))
SUITE["events"]["ode_division"] = BenchmarkGroup()
SUITE["events"]["ode_division"]["Direct"] = benchmark_solve(
    jump_problem(
        ODEProblem{true, FullSpecialize}(
            telegraph_growth_f!, [1.0, 0.0, 0.0], (0.0, 500.0)),
        Direct(), telegraph_maj; save_positions = (false, false)),
    Tsit5(); callback = division, save_everystep = false)

# Test 9 of test/variable_rate.jl, scaled up: u[1] decays and a ContinuousCallback resets it
# from 1/2 to 1, so its events cut solver steps short; a VariableRateJump with rate 100 u[1]
# counts in u[2]
vrj_reset_f!(du, u, p, t) = (du[1] = -u[1]; du[2] = 0; nothing)
vrj_reset_prob = ODEProblem{true, FullSpecialize}(vrj_reset_f!, [1.0, 0.0], (0.0, 10.0))
vrj_reset = ContinuousCallback((u, t, integrator) -> u[1] - 0.5,
    integrator -> (integrator.u[1] = 1.0; nothing); save_positions = (false, false))
vrj_reset_jump = VariableRateJump((u, p, t) -> 100 * u[1],
    integrator -> (integrator.u[2] += 1; nothing); save_positions = (false, false))
SUITE["events"]["vrj_reset"] = BenchmarkGroup()
for vr_aggregator in (VR_FRM(), VR_Direct())
    SUITE["events"]["vrj_reset"][aggname(vr_aggregator)] = benchmark_solve(
        jump_problem(vrj_reset_prob, vrj_reset_jump; vr_aggregator), Tsit5();
        callback = vrj_reset, save_everystep = false)
end

# The SDE telegraph gene with 20 proteins added at fixed times by a DiscreteCallback
const SDE_DOSING_TIMES = (25.0, 50.0, 75.0)
sde_dosing = DiscreteCallback((u, t, integrator) -> t in SDE_DOSING_TIMES,
    integrator -> (integrator.u[3] += 20; nothing); save_positions = (false, false))
SUITE["events"]["sde_dosing"] = BenchmarkGroup()
SUITE["events"]["sde_dosing"]["Direct"] = benchmark_solve(
    jump_problem(telegraph_sde, Direct(), telegraph_maj; save_positions = (false, false)),
    SRIW1(); callback = sde_dosing, tstops = collect(SDE_DOSING_TIMES),
    save_everystep = false)
