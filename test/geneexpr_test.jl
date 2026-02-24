using JumpProcesses, OrdinaryDiffEq
using Test, Statistics
using StableRNGs
rng = StableRNG(12345)

# using Plots; plotlyjs()
doplot = false
#using BenchmarkTools
# dobenchmark = false

dotestmean = true
doprintmeans = false

# SSAs to test
SSAalgs = (JumpProcesses.JUMP_AGGREGATORS..., JumpProcesses.NullAggregator())

# numerical parameters
Nsims = 8000
tf = 1000.0
u0 = [1, 0, 0, 0]
expected_avg = 5.926553750000000e+02
reltol = 0.01

# average number of proteins in a simulation
function runSSAs(jump_prob; use_stepper = true, rng = nothing)
    Psamp = zeros(Int, Nsims)
    for i in 1:Nsims
        sol = if use_stepper
            isnothing(rng) ? solve(jump_prob, SSAStepper()) :
            solve(jump_prob, SSAStepper(); rng)
        else
            isnothing(rng) ? solve(jump_prob) : solve(jump_prob; rng)
        end
        Psamp[i] = sol[3, end]
    end
    mean(Psamp)
end

function runSSAs_ode(vrjprob; rng = nothing)
    Psamp = zeros(Float64, Nsims)
    tsave = vrjprob.prob.tspan[2]
    integrator = if isnothing(rng)
        init(vrjprob, Tsit5(); saveat = tsave)
    else
        init(vrjprob, Tsit5(); saveat = tsave, rng)
    end
    solve!(integrator)
    Psamp[1] = integrator.sol[3, end]
    for i in 2:Nsims
        reinit!(integrator)
        solve!(integrator)
        Psamp[i] = integrator.sol[3, end]
    end
    return mean(Psamp)
end

# MODEL SETUP

# DNA repression model DiffEqBiological
# using DiffEqBiological
# rs = @reaction_network dtype begin
#     k1, DNA --> mRNA + DNA
#     k2, mRNA --> mRNA + P
#     k3, mRNA --> 0
#     k4, P --> 0
#     k5, DNA + P --> DNAR
#     k6, DNAR --> DNA + P
# end k1 k2 k3 k4 k5 k6

# model using mass action jumps
# ids: DNA=1, mRNA = 2, P = 3, DNAR = 4
reactstoch = [
    [1 => 1],
    [2 => 1],
    [2 => 1],
    [3 => 1],
    [1 => 1, 3 => 1],
    [4 => 1]
]
netstoch = [
    [2 => 1],
    [3 => 1],
    [2 => -1],
    [3 => -1],
    [1 => -1, 3 => -1, 4 => 1],
    [1 => 1, 3 => 1, 4 => -1]
]
spec_to_dep_jumps = [[1, 5], [2, 3], [4, 5], [6]]
jump_to_dep_specs = [[2], [3], [2], [3], [1, 3, 4], [1, 3, 4]]
rates = [0.5, (20 * log(2.0) / 120.0), (log(2.0) / 120.0), (log(2.0) / 600.0), 0.025, 1.0]
majumps = MassActionJump(rates, reactstoch, netstoch)

# TESTING:
prob = DiscreteProblem(u0, (0.0, tf), rates)

# floating point u0 tests
u0f = Float64.(u0)
probf = DiscreteProblem(u0f, (0.0, tf), rates)

# plotting one full trajectory
if doplot
    plothand = plot(reuse = false)
    for alg in SSAalgs
        local jump_prob = JumpProblem(prob, alg, majumps,
            vartojumps_map = spec_to_dep_jumps,
            jumptovars_map = jump_to_dep_specs)
        local sol = solve(jump_prob, SSAStepper(); rng)
        plot!(plothand, sol.t, sol[3, :], seriestype = :steppost)
    end
    display(plothand)
end

# test the means
if dotestmean
    for (i, alg) in enumerate(SSAalgs)
        local jump_prob = JumpProblem(prob, alg, majumps, save_positions = (false, false),
            vartojumps_map = spec_to_dep_jumps,
            jumptovars_map = jump_to_dep_specs)
        means = runSSAs(jump_prob; rng)
        relerr = abs(means - expected_avg) / expected_avg
        doprintmeans && println("Mean from method: ", typeof(alg), " is = ", means,
            ", rel err = ", relerr)
        @test abs(means - expected_avg) < reltol * expected_avg
    end

    # test default solver dispatch (use_stepper=false) with one algorithm
    let alg = Direct()
        jump_prob = JumpProblem(prob, alg, majumps, save_positions = (false, false),
            vartojumps_map = spec_to_dep_jumps,
            jumptovars_map = jump_to_dep_specs)
        means = runSSAs(jump_prob; use_stepper = false, rng)
        @test abs(means - expected_avg) < reltol * expected_avg
    end

    # test Float64 u0 with Direct and RSSA (RSSA needs explicit testing due to
    # past rejection bound issues with floating point)
    for alg in (Direct(), RSSA())
        jump_probf = JumpProblem(probf, alg, majumps, save_positions = (false, false),
            vartojumps_map = spec_to_dep_jumps,
            jumptovars_map = jump_to_dep_specs)
        means = runSSAs(jump_probf; rng)
        relerr = abs(means - expected_avg) / expected_avg
        doprintmeans && println("Mean from method (Float64 u0): ", typeof(alg),
            " is = ", means, ", rel err = ", relerr)
        @test abs(means - expected_avg) < reltol * expected_avg
    end
end

# no-aggregator tests
jump_prob = JumpProblem(prob, majumps; save_positions = (false, false),
    vartojumps_map = spec_to_dep_jumps, jumptovars_map = jump_to_dep_specs)
@test abs(runSSAs(jump_prob; rng) - expected_avg) < reltol * expected_avg

jump_prob = JumpProblem(prob, majumps, save_positions = (false, false))
@test abs(runSSAs(jump_prob; rng) - expected_avg) < reltol * expected_avg

# crj/vrj accuracy test
#     k1, DNA --> mRNA + DNA
#     k2, mRNA --> mRNA + P
#     k3, mRNA --> 0
#     k4, P --> 0
#     k5, DNA + P --> DNAR
#     k6, DNAR --> DNA + P
#    DNA = 1, mRNA = 2, P = 3, DNAR = 4
let
    r1(u, p, t) = p[1] * u[1]
    r2(u, p, t) = p[2] * u[2]
    r3(u, p, t) = p[3] * u[2]
    r4(u, p, t) = p[4] * u[3]
    r5(u, p, t) = p[5] * u[1] * u[3]
    r6(u, p, t) = p[6] * u[4]
    a1!(integ) = (integ.u[2] += 1; nothing)
    a2!(integ) = (integ.u[3] += 1; nothing)
    a3!(integ) = (integ.u[2] -= 1; nothing)
    a4!(integ) = (integ.u[3] -= 1; nothing)
    function a5!(integ)
        integ.u[1] -= 1
        integ.u[3] -= 1
        integ.u[4] += 1
        nothing
    end
    function a6!(integ)
        integ.u[1] += 1
        integ.u[3] += 1
        integ.u[4] -= 1
        nothing
    end
    crjs = JumpSet(ConstantRateJump(r1, a1!), ConstantRateJump(r2, a2!),
        ConstantRateJump(r3, a3!), ConstantRateJump(r4, a4!), ConstantRateJump(r5, a5!),
        ConstantRateJump(r6, a6!))
    vrjs = JumpSet(VariableRateJump(r1, a1!; save_positions = (false, false)),
        VariableRateJump(r2, a2!, save_positions = (false, false)),
        VariableRateJump(r3, a3!, save_positions = (false, false)),
        VariableRateJump(r4, a4!, save_positions = (false, false)),
        VariableRateJump(r5, a5!, save_positions = (false, false)),
        VariableRateJump(r6, a6!, save_positions = (false, false)))

    prob = DiscreteProblem(u0, (0.0, tf), rates)
    crjprob = JumpProblem(prob, crjs; save_positions = (false, false))
    @test abs(runSSAs(crjprob; rng) - expected_avg) < reltol * expected_avg

    # vrjs are very slow so test on a shorter time span and compare to the crjs
    prob = DiscreteProblem(u0, (0.0, tf / 5), rates)
    crjprob = JumpProblem(prob, crjs; save_positions = (false, false))
    crjmean = runSSAs(crjprob; rng)
    f(du, u, p, t) = (du .= 0; nothing)
    oprob = ODEProblem(f, u0f, (0.0, tf / 5), rates)

    for vr_agg in (VR_FRM(), VR_Direct(), VR_DirectFW())
        vrjprob = JumpProblem(
            oprob, vrjs; vr_aggregator = vr_agg, save_positions = (false, false))
        vrjmean = runSSAs_ode(vrjprob; rng)
        @test abs(vrjmean - crjmean) < reltol * crjmean
    end
end
