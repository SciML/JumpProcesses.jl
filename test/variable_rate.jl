using DiffEqBase, JumpProcesses, OrdinaryDiffEq, SciMLBase, StochasticDiffEq, Test
using Random, LinearSolve, Statistics
using StableRNGs, ADTypes
rng = StableRNG(12345)

a = ExtendedJumpArray(rand(rng, 3), rand(rng, 2))
b = ExtendedJumpArray(rand(rng, 3), rand(rng, 2))

a .= b

@test a.u == b.u
@test a.jump_u == b.jump_u
@test a == b

c = rand(rng, 5)
d = 2.0

a .+ d
a .= b .+ d
a .+ c .+ d
a .= b .+ c .+ d

rate = (u, p, t) -> u[1]
affect! = (integrator) -> (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate, affect!)
jump2 = deepcopy(jump)

f = function (du, u, p, t)
    du[1] = u[1]
end

prob = ODEProblem(f, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM())
integrator = init(jump_prob, Tsit5(); rng)
sol = solve(jump_prob, Tsit5(); rng)
sol = solve(jump_prob, Rosenbrock23(autodiff = AutoFiniteDiff()); rng)
sol = solve(jump_prob, Rosenbrock23(); rng)

function vrfrm_trajectory(default_seed)
    task = @task begin
        Random.seed!(default_seed)
        local_rng = StableRNG(12345)
        ode_prob = ODEProblem((du, u, p, t) -> fill!(du, 0), [0.0], (0.0, 1.0))
        jump1 = VariableRateJump((u, p, t) -> 1.0,
            integrator -> (integrator.u[1] += 1))
        jump2 = VariableRateJump((u, p, t) -> 10.0,
            integrator -> (integrator.u[1] += 1))
        jump_prob = JumpProblem(ode_prob, jump1, jump2; vr_aggregator = VR_FRM())
        solve(jump_prob, Tsit5(); saveat = 0.0:0.05:1.0, rng = local_rng).u
    end
    schedule(task)
    return fetch(task)
end

@test vrfrm_trajectory(123) == vrfrm_trajectory(456)

jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct())
integrator = init(jump_prob_gill, Tsit5(); rng)
sol_gill = solve(jump_prob_gill, Tsit5(); rng)
sol_gill = solve(jump_prob, Rosenbrock23(autodiff = AutoFiniteDiff()))
sol_gill = solve(jump_prob, Rosenbrock23())
@test maximum([sol.u[i][2] for i in 1:length(sol.u)]) <= 1e-12
@test maximum([sol.u[i][3] for i in 1:length(sol.u)]) <= 1e-12

g = function (du, u, p, t)
    du[1] = u[1]
end
prob = SDEProblem(f, g, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM())
sol = solve(jump_prob, SRIW1(); rng)
jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct())
sol_gill = solve(jump_prob_gill, SRIW1(); rng)
@test maximum([sol.u[i][2] for i in 1:length(sol.u)]) <= 1e-12
@test maximum([sol.u[i][3] for i in 1:length(sol.u)]) <= 1e-12

function ff(du, u, p, t)
    if p == 0
        du .= 1.01u
    else
        du .= 2.01u
    end
end
function gg(du, u, p, t)
    du[1, 1] = 0.3u[1]
    du[1, 2] = 0.6u[1]
    du[2, 1] = 1.2u[1]
    du[2, 2] = 0.2u[2]
end
rate_switch(u, p, t) = u[1] * 1.0
function affect_switch!(integrator)
    integrator.p = 1
end
jump_switch = VariableRateJump(rate_switch, affect_switch!)
prob = SDEProblem(ff, gg, ones(2), (0.0, 1.0), 0, noise_rate_prototype = zeros(2, 2))
jump_prob = JumpProblem(prob, jump_switch; vr_aggregator = VR_FRM())
jump_prob_gill = JumpProblem(prob, jump_switch; vr_aggregator = VR_Direct())
sol = solve(jump_prob, SRA1(), dt = 1.0; rng)
sol_gill = solve(jump_prob_gill, SRA1(), dt = 1.0; rng)

## Some integration tests

function f2(du, u, p, t)
    du[1] = u[1]
end
prob = ODEProblem(f2, [0.2], (0.0, 10.0))
rate2(u, p, t) = 2
affect2!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = ConstantRateJump(rate2, affect2!)
jump_prob = JumpProblem(prob, jump; vr_aggregator = VR_FRM())
jump_prob_gill = JumpProblem(prob, jump; vr_aggregator = VR_Direct())
sol = solve(jump_prob, Tsit5(); rng)
sol_gill = solve(jump_prob_gill, Tsit5(); rng)
sol(4.0)
sol.u[4]

rate2b(u, p, t) = u[1]
affect2b!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate2b, affect2b!)
jump2 = deepcopy(jump)
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM())
jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct())
sol = solve(jump_prob, Tsit5(); rng)
sol_gill = solve(jump_prob_gill, Tsit5(); rng)
sol(4.0)
sol.u[4]

function g2(du, u, p, t)
    du[1] = u[1]
end
prob = SDEProblem(f2, g2, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM())
jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct())
sol = solve(jump_prob, SRIW1(); rng)
sol_gill = solve(jump_prob_gill, SRIW1(); rng)
sol(4.0)
sol.u[4]

function f3(du, u, p, t)
    du .= u
end
prob = ODEProblem(f3, [1.0 2.0; 3.0 4.0], (0.0, 1.0))
rate3(u, p, t) = u[1] + u[2]
function affect3!(integrator)
    (integrator.u[1] = 0.25;
        integrator.u[2] = 0.5;
        integrator.u[3] = 0.75;
        integrator.u[4] = 1)
end
jump = VariableRateJump(rate3, affect3!)
jump_prob = JumpProblem(prob, jump; vr_aggregator = VR_FRM())
jump_prob_gill = JumpProblem(prob, jump; vr_aggregator = VR_Direct())
sol = solve(jump_prob, Tsit5(); rng)
sol_gill = solve(jump_prob_gill, Tsit5(); rng)

# test for https://discourse.julialang.org/t/differentialequations-jl-package-variable-rate-jumps-with-complex-variables/80366/2
function f4(dx, x, p, t)
    dx[1] = x[1]
end
rate4(x, p, t) = t
function affect4!(integrator)
    integrator.u[1] = integrator.u[1] * 0.5
end
jump = VariableRateJump(rate4, affect4!)
x₀ = 1.0 + 0.0im
Δt = (0.0, 6.0)
prob = ODEProblem(f4, [x₀], Δt)
jump_prob = JumpProblem(prob, jump; vr_aggregator = VR_FRM())
jump_prob_gill = JumpProblem(prob, jump; vr_aggregator = VR_Direct())
sol = solve(jump_prob, Tsit5(); rng)
sol_gill = solve(jump_prob_gill, Tsit5(); rng)

# Out of place test
drift(x, p, t) = p * x
rate2c(x, p, t) = 3 * max(0.0, x[1])
affect!2(integrator) = (integrator.u ./= 2; nothing)
x0 = rand(2)
prob = ODEProblem(drift, x0, (0.0, 10.0), 2.0)
jump = VariableRateJump(rate2c, affect!2)
jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VR_FRM())

# test to check lack of dependency graphs is caught in Coevolve for systems with non-maj
# jumps
let
    maj_rate = [1.0]
    react_stoich_ = [Vector{Pair{Int, Int}}()]
    net_stoich_ = [[1 => 1]]
    mass_action_jump_ = MassActionJump(maj_rate, react_stoich_, net_stoich_;
        scale_rates = false)

    affect! = function (integrator)
        integrator.u[1] -= 1
    end
    cs_rate1(u, p, t) = 0.2 * u[1]
    constant_rate_jump = ConstantRateJump(cs_rate1, affect!)
    jumpset_ = JumpSet((), (constant_rate_jump,), nothing, mass_action_jump_)

    for alg in (Coevolve(),)
        u0 = [0]
        tspan = (0.0, 30.0)
        dprob_ = DiscreteProblem(u0, tspan)
        @test_throws ErrorException JumpProblem(dprob_, alg, jumpset_,
            save_positions = (false, false))

        vrj = VariableRateJump(cs_rate1, affect!; urate = ((u, p, t) -> 1.0),
            rateinterval = ((u, p, t) -> 1.0))
        @test_throws ErrorException JumpProblem(dprob_, alg, mass_action_jump_, vrj;
            save_positions = (false, false))
    end
end

# Test that rate, urate and lrate do not get called past tstop
# https://github.com/SciML/JumpProcesses.jl/issues/330
let
    function test_rate(u, p, t)
        if t > 1.0
            error("test_rate does not handle t > 1.0")
        else
            return 0.1
        end
    end
    test_affect!(integrator) = (integrator.u[1] += 1)
    function test_lrate(u, p, t)
        if t > 1.0
            error("test_lrate does not handle t > 1.0")
        else
            return 0.05
        end
    end
    function test_urate(u, p, t)
        if t > 1.0
            error("test_urate does not handle t > 1.0")
        else
            return 0.2
        end
    end

    test_jump = VariableRateJump(test_rate, test_affect!; urate = test_urate,
        rateinterval = (u, p, t) -> 1.0)

    dprob = DiscreteProblem([0], (0.0, 1.0), nothing)
    jprob = JumpProblem(dprob, Coevolve(), test_jump; dep_graph = [[1]])

    @test_nowarn for i in 1:50
        solve(jprob, SSAStepper())
    end
end

# Coevolve in continuous integrators must test candidates tied with the current time, such as
# the shared `rateinterval` ends of jumps whose bounds were computed together. A tstop at the
# current time is never reached, so a tied candidate used to stall every Coevolve jump.
let
    f!(du, u, p, t) = (du .= 0; nothing)
    g!(du, u, p, t) = (du .= 0; nothing)
    rate(u, p, t) = 1.0
    urate(u, p, t) = 1.0
    rateinterval(u, p, t) = 0.1
    jump1 = VariableRateJump(rate, integ -> (integ.u[1] += 1); urate, rateinterval)
    jump2 = VariableRateJump(rate, integ -> (integ.u[2] += 1); urate, rateinterval)
    u0 = [0.0, 0.0]
    tspan = (0.0, 10.0)
    Nsims = 1000
    for (prob, alg) in ((ODEProblem(f!, u0, tspan), Tsit5()),
        (SDEProblem(f!, g!, u0, tspan), SRIW1()))
        jprob = JumpProblem(prob, Coevolve(), jump1, jump2; dep_graph = [[1], [2]])
        counts = [solve(jprob, alg; rng, save_everystep = false).u[end] for _ in 1:Nsims]
        # each jump fires Poisson(10) times
        @test mean(c -> c[1], counts)≈10 atol=0.5
        @test mean(c -> c[2], counts)≈10 atol=0.5
    end
end

# A `rateinterval` that ends at the end time, as `1 - t` does on (0, 1), must not be renewed
# there: its window would have zero length, so testing tied candidates would never stop.
let
    f!(du, u, p, t) = (du .= 0; nothing)
    g!(du, u, p, t) = (du .= 0; nothing)
    rate(u, p, t) = 1.0
    urate(u, p, t) = 1.0
    rateinterval(u, p, t) = 1.0 - t
    jump = VariableRateJump(rate, integ -> (integ.u[1] += 1); urate, rateinterval)
    tspan = (0.0, 1.0)
    Nsims = 1000
    for (prob, alg) in ((ODEProblem(f!, [0.0], tspan), Tsit5()),
        (SDEProblem(f!, g!, [0.0], tspan), SRIW1()))
        jprob = JumpProblem(prob, Coevolve(), jump; dep_graph = [[1]])
        counts = [solve(jprob, alg; rng, save_everystep = false).u[end][1] for _ in 1:Nsims]
        # the jump fires Poisson(1) times
        @test mean(counts)≈1 atol=0.15
    end
end

# test u0 resets correctly
let
    b = 2.0
    d = 1.0
    n0 = 1
    tspan = (0.0, 4.0)
    Nsims = 10
    u0 = [n0]
    p = [b, d]

    function ode_fxn(du, u, p, t)
        du .= 0
        nothing
    end
    b_rate(u, p, t) = (u[1] * p[1])
    function birth!(integrator)
        integrator.u[1] += 1
        nothing
    end
    b_jump = VariableRateJump(b_rate, birth!)

    d_rate(u, p, t) = (u[1] * p[2])
    function death!(integrator)
        integrator.u[1] -= 1
        nothing
    end
    d_jump = VariableRateJump(d_rate, death!)

    ode_prob = ODEProblem(ode_fxn, u0, tspan, p)
    sjm_prob = JumpProblem(ode_prob, b_jump, d_jump; vr_aggregator = VR_FRM())
    # After callback initialize, integrator.u.jump_u should have unique thresholds
    # that differ between sequential solves (RNG advances each time).
    jump_u_old = zeros(length(sjm_prob.prob.u0.jump_u))
    for i in 1:Nsims
        integrator = init(sjm_prob, Tsit5(); saveat = tspan[2], rng)
        @test allunique(integrator.u.jump_u)
        @test integrator.u.jump_u != jump_u_old
        jump_u_old .= integrator.u.jump_u
        solve!(integrator)
    end
end

# accuracy test based on 
# https://github.com/SciML/JumpProcesses.jl/issues/320
function getmean(Nsims, prob, alg, tsave, rng)
    umean = zeros(length(tsave))
    integrator = init(prob, alg; saveat = tsave, rng)
    solve!(integrator)
    for j in eachindex(umean)
        umean[j] += integrator.sol.u[j][1]
    end
    for i in 2:Nsims
        reinit!(integrator)
        solve!(integrator)
        for j in eachindex(umean)
            umean[j] += integrator.sol.u[j][1]
        end
    end
    umean ./= Nsims
    return umean
end

let
    rng_seed = 12345
    b = 2.0
    d = 1.0
    n0 = 1.0
    tspan = (0.0, 4.0)
    Nsims = 10000
    n(t) = n0 * exp((b - d) * t)
    u0 = [n0]
    p = [b, d]

    function ode_fxn(du, u, p, t)
        du .= 0
        nothing
    end

    b_rate(u, p, t) = (u[1] * p[1])
    function birth!(integrator)
        integrator.u[1] += 1
        nothing
    end
    b_jump = VariableRateJump(b_rate, birth!; save_positions = (false, false))

    d_rate(u, p, t) = (u[1] * p[2])
    function death!(integrator)
        integrator.u[1] -= 1
        nothing
    end
    d_jump = VariableRateJump(d_rate, death!; save_positions = (false, false))

    ode_prob = ODEProblem(ode_fxn, u0, tspan, p)
    tsave = range(tspan[1], tspan[2]; step = 0.1)
    for vr_aggregator in (VR_Direct(), VR_DirectFW(), VR_FRM(),
        VR_FRM(; vector_callback = true))
        sjm_prob = JumpProblem(ode_prob, b_jump, d_jump; vr_aggregator)

        for alg in (Tsit5(), Rodas5P(linsolve = QRFactorization()))
            umean = getmean(Nsims, sjm_prob, alg, tsave, StableRNG(rng_seed))
            @test all(abs.(umean .- n.(tsave)) .< 0.05 * n.(tsave))
            rng_seed += Nsims
        end
    end
end

# Correctness test based on 
# VR_Direct and VR_FRM
# Function to run ensemble and compute statistics
function run_ensemble(prob, alg, jumps...; vr_aggregator = VR_FRM(), Nsims = 8000)
    rng = StableRNG(12345)
    jump_prob = JumpProblem(prob, Direct(), jumps...; vr_aggregator)
    total = 0.0
    for i in 1:Nsims
        sol = solve(jump_prob, alg; save_everystep = false, rng)
        total += sol.u[end][1]
    end
    return total / Nsims
end

# Test 1: Simple ODE with two variable rate jumps
let
    rate = (u, p, t) -> u[1]
    affect! = (integrator) -> (integrator.u[1] = integrator.u[1] / 2)
    jump = VariableRateJump(rate, affect!; save_positions = (false, false))
    jump2 = deepcopy(jump)

    f = (du, u, p, t) -> (du[1] = u[1])
    prob = ODEProblem(f, [0.2], (0.0, 10.0))

    mean_vrfr = run_ensemble(prob, Tsit5(), jump, jump2)
    mean_vrcb = run_ensemble(prob, Tsit5(), jump, jump2; vr_aggregator = VR_Direct())
    mean_vrcbfw = run_ensemble(prob, Tsit5(), jump, jump2; vr_aggregator = VR_DirectFW())

    @test isapprox(mean_vrfr, mean_vrcb, rtol = 0.05)
    @test isapprox(mean_vrcb, mean_vrcbfw, rtol = 0.05)
end

# Test 2: SDE with two variable rate jumps
let
    f = (du, u, p, t) -> (du[1] = -u[1] / 10.0)
    g = (du, u, p, t) -> (du[1] = -u[1] / 10.0)
    rate = (u, p, t) -> u[1] / 10.0
    affect! = (integrator) -> (integrator.u[1] = integrator.u[1] + 1)
    jump = VariableRateJump(rate, affect!; save_positions = (false, false))
    jump2 = deepcopy(jump)

    prob = SDEProblem(f, g, [10.0], (0.0, 10.0))

    mean_vrfr = run_ensemble(prob, SRIW1(), jump, jump2)
    mean_vrcb = run_ensemble(prob, SRIW1(), jump, jump2; vr_aggregator = VR_Direct())
    mean_vrcbfw = run_ensemble(prob, SRIW1(), jump, jump2; vr_aggregator = VR_DirectFW())

    @test isapprox(mean_vrfr, mean_vrcb, rtol = 0.05)
    @test isapprox(mean_vrcb, mean_vrcbfw, rtol = 0.05)
end

# Test 3: ODE with analytical solution
let
    λ = 2.0
    f = (du, u, p, t) -> (du[1] = -u[1]; nothing)
    rate = (u, p, t) -> λ
    affect! = (integrator) -> (integrator.u[1] += 1; nothing)
    jump = VariableRateJump(rate, affect!; save_positions = (false, false))

    prob = ODEProblem(f, [0.2], (0.0, 10.0))

    mean_vrfr = run_ensemble(prob, Tsit5(), jump)
    mean_vrcb = run_ensemble(prob, Tsit5(), jump; vr_aggregator = VR_Direct())
    mean_vrcbfw = run_ensemble(prob, Tsit5(), jump; vr_aggregator = VR_DirectFW())

    t = 10.0
    u0 = 0.2
    analytical_mean = u0 * exp(-t) + λ*(1 - exp(-t))

    @test isapprox(mean_vrfr, analytical_mean, rtol = 0.05)
    @test isapprox(mean_vrfr, mean_vrcb, rtol = 0.05)
    @test isapprox(mean_vrcb, mean_vrcbfw, rtol = 0.05)
end

# Test 4: No. of Jumps
let
    f(du, u, p, t) = (du[1] = 0.0; nothing)

    # Define birth jump: ∅ → X
    birth_rate(u, p, t) = 10.0
    function birth_affect!(integrator)
        integrator.u[1] += 1
        integrator.p[3] += 1
        nothing
    end
    birth_jump = VariableRateJump(birth_rate, birth_affect!; save_positions = (
        false, false))

    # Define death jump: X → ∅
    death_rate(u, p, t) = 0.5 * u[1]
    function death_affect!(integrator)
        integrator.u[1] -= 1
        integrator.p[3] += 1
        nothing
    end
    death_jump = VariableRateJump(death_rate, death_affect!; save_positions = (
        false, false))

    Nsims = 100
    results = Dict()
    u0 = [1.0]
    tspan = (0.0, 10.0)
    for vr_aggregator in (VR_FRM(), VR_Direct(), VR_DirectFW())
        jump_counts = zeros(Int, Nsims)
        p = [0.0, 0.0, 0]
        prob = ODEProblem(f, u0, tspan, p)
        jump_prob = JumpProblem(prob, Direct(), birth_jump, death_jump; vr_aggregator)

        for i in 1:Nsims
            sol = solve(jump_prob, Tsit5(); save_everystep = false, rng)
            jump_counts[i] = jump_prob.prob.p[3]
            jump_prob.prob.p[3] = 0
        end

        results[vr_aggregator] = (mean_jumps = mean(jump_counts), jump_counts = jump_counts)
        @test sum(jump_counts) > 1000
    end

    mean_jumps_vrfr = results[VR_FRM()].mean_jumps
    mean_jumps_vrcb = results[VR_Direct()].mean_jumps
    mean_jumps_vrcbfw = results[VR_DirectFW()].mean_jumps

    @test isapprox(mean_jumps_vrfr, mean_jumps_vrcb, rtol = 0.1)
    @test isapprox(mean_jumps_vrcb, mean_jumps_vrcbfw, rtol = 0.1)
end

# Test 5: Time span that doesn't start at zero
let
    λ = 0.5
    f = (du, u, p, t) -> (du[1] = 0.0; nothing)
    rate = (u, p, t) -> λ
    affect! = (integrator) -> (integrator.u[1] += 1; nothing)
    jump = VariableRateJump(rate, affect!; save_positions = (false, false))

    prob = ODEProblem(f, [0.0], (5.0, 15.0))
    analytical_mean = λ * 10.0
    for vr_aggregator in (VR_FRM(), VR_Direct(), VR_DirectFW())
        mean_jumps = run_ensemble(prob, Tsit5(), jump; vr_aggregator)
        @test isapprox(mean_jumps, analytical_mean, rtol = 0.05)
    end
end

# Test 6: Rates that vary within the solver's steps. The ODE part is constant, so it doesn't
# limit the step size, and VR_Direct's integral of the total rate must resolve the rates'
# variation itself. This tests the fix for
# https://github.com/SciML/JumpProcesses.jl/issues/692
let
    λ = 0.1
    f = (du, u, p, t) -> (du[1] = 0.0; nothing)
    affect! = (integrator) -> (integrator.u[1] += 1; nothing)
    prob = ODEProblem(f, [0.0], (0.0, 10.0))

    # a rate that oscillates throughout, and one that starts oscillating at t = 5
    rates = ((u, p, t) -> λ * (1 + sin(10 * t)),
        (u, p, t) -> t < 5 ? λ : λ * (1 + sin(10 * (t - 5))))
    analytical_means = (λ * (10 + (1 - cos(100)) / 10), λ * (10 + (1 - cos(50)) / 10))
    for (rate, analytical_mean) in zip(rates, analytical_means)
        jump = VariableRateJump(rate, affect!; save_positions = (false, false))
        for vr_aggregator in (VR_FRM(), VR_Direct(), VR_DirectFW())
            mean_jumps = run_ensemble(prob, Tsit5(), jump; vr_aggregator)
            @test isapprox(mean_jumps, analytical_mean, rtol = 0.05)
        end
    end
end

# Test combining MassActionJump with VariableRateJump for all aggregators
# This tests the fix for https://github.com/SciML/JumpProcesses.jl/issues/545
# where RSSA/RSSACR aggregators would fail with ExtendedJumpArray due to
# update_u_brackets! iterating over the full array instead of just species
let
    rng = StableRNG(12345)

    function f!(du, u, p, t)
        du .= 0
    end

    # VariableRateJump (rate depends on time)
    vrj_rate(u, p, t) = p[1] * (1 + sin(t)) * u[1]
    vrj_affect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
    vrj = VariableRateJump(vrj_rate, vrj_affect!)

    # MassActionJump: X2 -> X1 with rate k2
    maj = MassActionJump([0.5], [[2 => 1]], [[1 => 1, 2 => -1]])

    u0 = [10.0, 5.0]
    tspan = (0.0, 1.0)
    p = [0.1]
    prob = ODEProblem(f!, u0, tspan, p)

    # Test all aggregators that support MassActionJump
    # Note: Coevolve requires dependency graphs for non-bounded VRJs so we skip it here
    aggregators = [
        Direct(),
        FRM(),
        NRM(),
        SortingDirect(),
        DirectCR(),
        RDirect(),
        RSSA(),
        RSSACR()
    ]

    for agg in aggregators
        local jprob = JumpProblem(prob, agg, maj, vrj)
        local sol = solve(jprob, Tsit5(); rng)
        @test SciMLBase.successful_retcode(sol)
        # Verify conservation: total population should be conserved
        @test sol.u[end][1] + sol.u[end][2] ≈ u0[1] + u0[2]
    end
end

# Test with ConstantRateJump + VariableRateJump (no MassActionJump)
let
    rng = StableRNG(12345)

    function f!(du, u, p, t)
        du .= 0
    end

    # VariableRateJump
    vrj_rate(u, p, t) = 0.1 * (1 + sin(t)) * u[1]
    vrj_affect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
    vrj = VariableRateJump(vrj_rate, vrj_affect!)

    # ConstantRateJump: X2 -> X1
    crj_rate(u, p, t) = 0.5 * u[2]
    crj_affect!(integrator) = (integrator.u[1] += 1; integrator.u[2] -= 1)
    crj = ConstantRateJump(crj_rate, crj_affect!)

    u0 = [10.0, 5.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(f!, u0, tspan)

    # Test with Direct aggregator (most common case)
    jprob = JumpProblem(prob, Direct(), crj, vrj)
    sol = solve(jprob, Tsit5(); rng)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[end][1] + sol.u[end][2] ≈ u0[1] + u0[2]
end

# Test with multiple VariableRateJumps + MassActionJump
let
    rng = StableRNG(12345)

    function f!(du, u, p, t)
        du .= 0
    end

    # Two VariableRateJumps
    vrj1_rate(u, p, t) = 0.1 * u[1]
    vrj1_affect!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1)
    vrj1 = VariableRateJump(vrj1_rate, vrj1_affect!)

    vrj2_rate(u, p, t) = 0.05 * u[2]
    vrj2_affect!(integrator) = (integrator.u[2] -= 1; integrator.u[1] += 1)
    vrj2 = VariableRateJump(vrj2_rate, vrj2_affect!)

    # MassActionJump
    maj = MassActionJump([0.2], [[1 => 1]], [[1 => -1, 2 => 1]])

    u0 = [20.0, 10.0]
    tspan = (0.0, 2.0)
    prob = ODEProblem(f!, u0, tspan)

    # Test RSSA and RSSACR specifically (the aggregators that had the bug)
    for agg in [RSSA(), RSSACR()]
        local jprob = JumpProblem(prob, agg, maj, vrj1, vrj2)
        local sol = solve(jprob, Tsit5(); rng)
        @test SciMLBase.successful_retcode(sol)
        @test sol.u[end][1] + sol.u[end][2] ≈ u0[1] + u0[2]
    end
end

# Regression for https://github.com/SciML/JumpProcesses.jl/issues/592
# VR_FRM + SDEProblem with `noise_rate_prototype` + adaptive non-diagonal SDE
# solver: ExtendedJumpArray's `mul!` needs to zero `c.jump_u` when the noise
# matrix only addresses the original state, otherwise stale scratchpad values
# from the adaptive error estimate blow up `jump_u` and the VRJ callbacks
# never fire.
let
    rng = StableRNG(592)

    mutable struct Issue592Params
        λ::Float64
        μ::Float64
        mode::Symbol
    end
    p = Issue592Params(0.3, 0.5, :CLE)

    function f!(du, u, p, t)
        du[1] = (p.mode === :CLE) ? (p.λ - p.μ) * u[1] : 0.0
        nothing
    end
    function g!(G, u, p, t)
        if p.mode === :CLE
            x = max(u[1], 0.0)
            G[1, 1] = sqrt(p.λ * x)
            G[1, 2] = -sqrt(p.μ * x)
        else
            G .= 0.0
        end
        nothing
    end

    rate_birth(u, p, t) = (p.mode === :SSA) ? p.λ * u[1] : 0.0
    rate_death(u, p, t) = (p.mode === :SSA) ? p.μ * u[1] : 0.0
    birth_affect!(integ) = (integ.u[1] += 1.0)
    death_affect!(integ) = (integ.u[1] -= 1.0)
    birth = VariableRateJump(rate_birth, birth_affect!)
    death = VariableRateJump(rate_death, death_affect!)

    switch_cond(u, t, integ) = (integ.p.mode === :CLE) ? u[1] - 95.0 : 1.0
    function switch_affect!(integ)
        integ.u[1] = max(round(integ.u[1]), 0.0)
        integ.p.mode = :SSA
        derivative_discontinuity!(integ, true)
        reset_aggregated_jumps!(integ)
    end
    switch_cb = ContinuousCallback(switch_cond, switch_affect!)

    u0 = [150.0]
    tspan = (0.0, 10.0)
    sde_prob = SDEProblem(f!, g!, u0, tspan, p; noise_rate_prototype = zeros(1, 2))
    jprob = JumpProblem(sde_prob, Direct(), birth, death; vr_aggregator = VR_FRM())

    sol = solve(jprob, LambaEM(); rng, callback = switch_cb, adaptive = true)
    @test SciMLBase.successful_retcode(sol)
    @test sol.t[end] == tspan[2]
    # The bug drove `jump_u` past ±1e15 within a few steps; with the fix it
    # stays bounded near its initial -randexp() values.
    @test all(isfinite, sol.u[end].jump_u)
    @test maximum(abs, sol.u[end].jump_u) < 1e6
    # SSA jumps must actually fire after the mode switch — without the fix u
    # gets pinned at the switch threshold (95) forever.
    @test sol.u[end].u[1] != 95.0
end

# `VR_FRM(; vector_callback = true)` handles the jumps with one `VectorContinuousCallback` per
# set of callback options, and `VR_FRM(; vector_callback = false)` with one
# `ContinuousCallback` per jump. `VR_FRM()` uses the latter for up to three jumps. Both draw
# the same random numbers in the same order, so a seed gives the same trajectory.
@testset "VR_FRM vector and per-jump callbacks" begin
    @test VR_FRM() === VR_FRM{nothing}()
    @test VR_FRM(; vector_callback = true) === VR_FRM{true}()
    @test VR_FRM(; vector_callback = false) === VR_FRM{false}()
    vector_agg = VR_FRM(; vector_callback = true)
    per_jump_agg = VR_FRM(; vector_callback = false)

    # u[1:N] count the firings of each jump, u[N + 1] = c(t) drives the rates
    function vr_frm_prob(N, vr_aggregator; sde = false, opts = i -> (;))
        f!(du, u, p, t) = (du .= 0; du[end] = -0.1 * u[end]; nothing)
        g!(du, u, p, t) = (du .= 0; du[end] = 0.05 * u[end]; nothing)
        u0 = [zeros(N); 1.0]
        prob = sde ? SDEProblem(f!, g!, u0, (0.0, 10.0), [2.0]) :
               ODEProblem(f!, u0, (0.0, 10.0), [2.0])
        jumps = [VariableRateJump((u, p, t) -> p[1] * u[end] * (1 + sin(t + i)),
                     integ -> (integ.u[i] += 1; nothing); opts(i)...) for i in 1:N]
        JumpProblem(prob, JumpSet(; variable_jumps = jumps); vr_aggregator)
    end
    # odd and even jumps get different callback options
    mixed = i -> isodd(i) ? (;) : (; save_positions = (false, false), interp_points = 5)

    # callbacks and the components of the extended state they interpolate
    ccs = vr_frm_prob(3, vector_agg).jump_callback.continuous_callbacks
    @test length(ccs) == 1
    @test ccs[1] isa VectorContinuousCallback
    @test ccs[1].idxs === 5:7
    ccs = vr_frm_prob(3, per_jump_agg).jump_callback.continuous_callbacks
    @test length(ccs) == 3
    @test all(cb -> cb isa ContinuousCallback, ccs)
    @test [cb.idxs for cb in ccs] == [5, 6, 7]
    ccs = vr_frm_prob(5, vector_agg; opts = mixed).jump_callback.continuous_callbacks
    @test length(ccs) == 2
    @test ccs[1].idxs == [7, 9, 11]
    @test ccs[2].idxs == [8, 10]
    @test ccs[2].interp_points == 5
    @test ccs[2].save_positions == [false, false]

    # `VR_FRM()` chooses by the number of jumps
    ccs = vr_frm_prob(3, VR_FRM()).jump_callback.continuous_callbacks
    @test length(ccs) == 3
    @test all(cb -> cb isa ContinuousCallback, ccs)
    ccs = vr_frm_prob(4, VR_FRM()).jump_callback.continuous_callbacks
    @test length(ccs) == 1
    @test ccs[1] isa VectorContinuousCallback
    @test ccs[1].idxs === 6:9

    # more than 32 jumps; non-contiguous `idxs` from the mixed options
    same = i -> (;)
    cases = ((3, false, Tsit5(), same),
        (40, false, Tsit5(), same),
        (3, false, Rodas5P(), same),
        (3, true, SRIW1(), same),
        (5, false, Tsit5(), mixed))
    for (N, sde, alg, opts) in cases
        sol = solve(vr_frm_prob(N, vector_agg; sde, opts), alg; rng = StableRNG(7))
        sol_pj = solve(vr_frm_prob(N, per_jump_agg; sde, opts), alg; rng = StableRNG(7))
        @test SciMLBase.successful_retcode(sol)
        @test all(>(0), sol.u[end].u[1:N])
        @test sol.t ≈ sol_pj.t
        @test sol.u[end] ≈ sol_pj.u[end]
    end

    # the callbacks index the extended state, so they reject a state of a different length
    for vr_aggregator in (vector_agg, per_jump_agg)
        jprob = vr_frm_prob(2, vr_aggregator)
        sol = solve(remake(jprob; u0 = [0.0, 0.0, 1.0]), Tsit5(); rng = StableRNG(1))
        @test SciMLBase.successful_retcode(sol)
        longer = remake(jprob; u0 = [0.0, 0.0, 0.0, 1.0])
        @test_throws ArgumentError solve(longer, Tsit5(); rng = StableRNG(1))
    end
end

# Mean jump counts against their exact values. Jump i has rate k c(t) (1 + sin(t + i)), where
# c' = -decay c, so its count over [0, T] has mean k ∫ exp(-decay t) (1 + sin(t + i)) dt. With
# two jumps `VR_FRM()` uses one `ContinuousCallback` per jump, and with six one
# `VectorContinuousCallback`; each size also forces the other kind. In the SDE problems c is a
# geometric Brownian motion, so E[c(t)], and with it each mean count, is unchanged.
@testset "VariableRateJump mean counts vs exact" begin
    decay, k, T = 0.1, 2.0, 10.0
    # an antiderivative of exp(-decay t) (1 + sin(t + i))
    F(t, i) = -exp(-decay * t) / decay +
              exp(-decay * t) * (-decay * sin(t + i) - cos(t + i)) / (1 + decay^2)
    exact_mean(i) = k * (F(T, i) - F(0.0, i))

    function counts_prob(N, vr_aggregator; sde = false, opts = i -> (;))
        f!(du, u, p, t) = (du .= 0; du[end] = -decay * u[end]; nothing)
        g!(du, u, p, t) = (du .= 0; du[end] = 0.1 * u[end]; nothing)
        u0 = [zeros(N); 1.0]
        prob = sde ? SDEProblem(f!, g!, u0, (0.0, T), [k]) :
               ODEProblem(f!, u0, (0.0, T), [k])
        jumps = [VariableRateJump((u, p, t) -> p[1] * u[end] * (1 + sin(t + i)),
                     integ -> (integ.u[i] += 1; nothing); opts(i)...) for i in 1:N]
        JumpProblem(prob, JumpSet(; variable_jumps = jumps); vr_aggregator)
    end
    # odd and even jumps get different callback options, so `VR_FRM` builds two callbacks
    mixed = i -> isodd(i) ? (;) : (; save_positions = (false, false), interp_points = 5)
    same = i -> (;)

    cases = Any[(N, agg, false, Tsit5(), same)
                for N in (2, 6)
                for agg in (VR_FRM(), VR_FRM(; vector_callback = N <= 3), VR_Direct(),
        VR_DirectFW())]
    push!(cases, (6, VR_FRM(), false, Tsit5(), mixed))
    push!(cases, (6, VR_FRM(), true, SRIW1(), same))
    push!(cases, (6, VR_Direct(), true, SRIW1(), same))

    Nsims = 4000
    for (N, vr_aggregator, sde, alg, opts) in cases
        jprob = counts_prob(N, vr_aggregator; sde, opts)
        rng = StableRNG(12345)
        counts = zeros(Nsims, N)
        for n in 1:Nsims
            sol = solve(jprob, alg; rng, save_everystep = false)
            for i in 1:N
                counts[n, i] = sol.u[end][i]
            end
        end
        z = [(mean(c) - exact_mean(i)) / sqrt(var(c) / Nsims)
             for (i, c) in enumerate(eachcol(counts))]
        @test all(<(4), abs.(z))
    end
end
