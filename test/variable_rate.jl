using DiffEqBase, JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Test
using Random, LinearSolve, Statistics
using StableRNGs
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
jump = VariableRateJump(rate, affect!, interp_points = 1000)
jump2 = deepcopy(jump)

f = function (du, u, p, t)
    du[1] = u[1]
end

prob = ODEProblem(f, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM(), rng)
integrator = init(jump_prob, Tsit5())
sol = solve(jump_prob, Tsit5())
sol = solve(jump_prob, Rosenbrock23(autodiff = false))
sol = solve(jump_prob, Rosenbrock23())

jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct(), rng)
integrator = init(jump_prob_gill, Tsit5())
sol_gill = solve(jump_prob_gill, Tsit5())
sol_gill = solve(jump_prob, Rosenbrock23(autodiff = false))
sol_gill = solve(jump_prob, Rosenbrock23())
@test maximum([sol.u[i][2] for i in 1:length(sol)]) <= 1e-12
@test maximum([sol.u[i][3] for i in 1:length(sol)]) <= 1e-12

g = function (du, u, p, t)
    du[1] = u[1]
end
prob = SDEProblem(f, g, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM(), rng = rng)
sol = solve(jump_prob,  SRIW1())
jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct(), rng=rng)
sol_gill = solve(jump_prob_gill,  SRIW1())
@test maximum([sol.u[i][2] for i in 1:length(sol)]) <= 1e-12
@test maximum([sol.u[i][3] for i in 1:length(sol)]) <= 1e-12

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
jump_prob = JumpProblem(prob, jump_switch; vr_aggregator = VR_FRM(), rng)
jump_prob_gill = JumpProblem(prob, jump_switch; vr_aggregator = VR_Direct(), rng)
sol = solve(jump_prob, SRA1(), dt = 1.0)
sol_gill = solve(jump_prob_gill, SRA1(), dt = 1.0)

## Some integration tests

function f2(du, u, p, t)
    du[1] = u[1]
end
prob = ODEProblem(f2, [0.2], (0.0, 10.0))
rate2(u, p, t) = 2
affect2!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = ConstantRateJump(rate2, affect2!)
jump_prob = JumpProblem(prob, jump; vr_aggregator = VR_FRM(), rng)
jump_prob_gill = JumpProblem(prob, jump; vr_aggregator = VR_Direct(), rng)
sol = solve(jump_prob, Tsit5())
sol_gill = solve(jump_prob_gill, Tsit5())
sol(4.0)
sol.u[4]

rate2b(u, p, t) = u[1]
affect2!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate2b, affect2!)
jump2 = deepcopy(jump)
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM(), rng)
jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct(), rng)
sol = solve(jump_prob, Tsit5())
sol_gill = solve(jump_prob_gill, Tsit5())
sol(4.0)
sol.u[4]

function g2(du, u, p, t)
    du[1] = u[1]
end
prob = SDEProblem(f2, g2, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, jump, jump2; vr_aggregator = VR_FRM(), rng)
jump_prob_gill = JumpProblem(prob, jump, jump2; vr_aggregator = VR_Direct(), rng)
sol = solve(jump_prob, SRIW1())
sol_gill = solve(jump_prob_gill, SRIW1())
sol(4.0)
sol.u[4]

function f3(du, u, p, t)
    du .= u
end
prob = ODEProblem(f3, [1.0 2.0; 3.0 4.0], (0.0, 1.0))
rate3(u, p, t) = u[1] + u[2]
affect3!(integrator) = (integrator.u[1] = 0.25;
integrator.u[2] = 0.5;
integrator.u[3] = 0.75;
integrator.u[4] = 1)
jump = VariableRateJump(rate3, affect3!)
jump_prob = JumpProblem(prob, jump; vr_aggregator = VR_FRM(), rng)
jump_prob_gill = JumpProblem(prob, jump; vr_aggregator = VR_Direct(), rng)
sol = solve(jump_prob, Tsit5())
sol_gill = solve(jump_prob_gill, Tsit5())

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
jump_prob = JumpProblem(prob, jump; vr_aggregator = VR_FRM(), rng)
jump_prob_gill = JumpProblem(prob, jump; vr_aggregator = VR_Direct(), rng)
sol = solve(jump_prob, Tsit5())
sol_gill = solve(jump_prob_gill, Tsit5())

# Out of place test
drift(x, p, t) = p * x
rate2c(x, p, t) = 3 * max(0.0, x[1])
affect!2(integrator) = (integrator.u ./= 2; nothing)
x0 = rand(2)
prob = ODEProblem(drift, x0, (0.0, 10.0), 2.0)
jump = VariableRateJump(rate2c, affect!2)
jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VR_FRM(), rng)

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
    sjm_prob = JumpProblem(ode_prob, b_jump, d_jump; vr_aggregator = VR_FRM(), rng)
    @test allunique(sjm_prob.prob.u0.jump_u)
    u0old = copy(sjm_prob.prob.u0.jump_u)
    for i in 1:Nsims
        sol = solve(sjm_prob, Tsit5(); saveat = tspan[2])
        @test allunique(sjm_prob.prob.u0.jump_u)
        @test all(u0old != sjm_prob.prob.u0.jump_u)
        u0old .= sjm_prob.prob.u0.jump_u
    end
end

# accuracy test based on 
# https://github.com/SciML/JumpProcesses.jl/issues/320
# note that even with the seeded StableRNG this test is not 
# deterministic for some reason.
function getmean(Nsims, prob, alg, dt, tsave, seed)
    umean = zeros(length(tsave))
    for i in 1:Nsims
        sol = solve(prob, alg; saveat = dt, seed)
        umean .+= Array(sol(tsave; idxs = 1))
        seed += 1
    end
    umean ./= Nsims
    return umean
end

let
    seed = 12345
    rng = StableRNG(seed)
    b = 2.0
    d = 1.0
    n0 = 1
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
    b_jump = VariableRateJump(b_rate, birth!)

    d_rate(u, p, t) = (u[1] * p[2])
    function death!(integrator)
        integrator.u[1] -= 1
        nothing
    end
    d_jump = VariableRateJump(d_rate, death!)

    ode_prob = ODEProblem(ode_fxn, u0, tspan, p)
    dt = 0.1
    tsave = range(tspan[1], tspan[2]; step = dt)
    for vr_aggregator in (VR_FRM(), VR_Direct())
        sjm_prob = JumpProblem(ode_prob, b_jump, d_jump; vr_aggregator, rng)

        for alg in (Tsit5(), Rodas5P(linsolve = QRFactorization()))
            umean = getmean(Nsims, sjm_prob, alg, dt, tsave, seed)
            @test all(abs.(umean .- n.(tsave)) .< 0.05 * n.(tsave))
            seed += Nsims 
        end
    end
end

# Correctness test based on 
# VR_Direct and VR_FRM
# Function to run ensemble and compute statistics
function run_ensemble(prob, alg, jumps...; vr_aggregator=VR_FRM(), Nsims=8000)
    rng = StableRNG(12345)
    jump_prob = JumpProblem(prob, Direct(), jumps...; vr_aggregator, rng)
    ensemble = EnsembleProblem(jump_prob)
    sol = solve(ensemble, alg, trajectories=Nsims, save_everystep=false)
    return mean(sol.u[i][1,end] for i in 1:Nsims)
end

# Test 1: Simple ODE with two variable rate jumps
let
    rate = (u, p, t) -> u[1]
    affect! = (integrator) -> (integrator.u[1] = integrator.u[1] / 2)
    jump = VariableRateJump(rate, affect!, interp_points=1000)
    jump2 = deepcopy(jump)
    
    f = (du, u, p, t) -> (du[1] = u[1])
    prob = ODEProblem(f, [0.2], (0.0, 10.0))
    
    mean_vrfr = run_ensemble(prob, Tsit5(), jump, jump2)
    mean_vrdcb = run_ensemble(prob, Tsit5(), jump, jump2; vr_aggregator=VR_Direct())
    
    @test isapprox(mean_vrfr, mean_vrdcb, rtol=0.05)
end

# Test 2: SDE with two variable rate jumps
let
    f = (du, u, p, t) -> (du[1] = -u[1] / 10.0)
    g = (du, u, p, t) -> (du[1] = -u[1] / 10.0)
    rate = (u, p, t) -> u[1] / 10.0
    affect! = (integrator) -> (integrator.u[1] = integrator.u[1] + 1)
    jump = VariableRateJump(rate, affect!)
    jump2 = deepcopy(jump)
    
    prob = SDEProblem(f, g, [10.0], (0.0, 10.0))
    
    mean_vrfr = run_ensemble(prob, SRIW1(), jump, jump2)
    mean_vrdcb = run_ensemble(prob, SRIW1(), jump, jump2; vr_aggregator=VR_Direct())
    
    @test isapprox(mean_vrfr, mean_vrdcb, rtol=0.05)
end

# Test 3: ODE with analytical solution
let
    λ = 2.0
    f = (du, u, p, t) -> (du[1] = -u[1]; nothing)
    rate = (u, p, t) -> λ
    affect! = (integrator) -> (integrator.u[1] += 1; nothing)
    jump = VariableRateJump(rate, affect!)
    
    prob = ODEProblem(f, [0.2], (0.0, 10.0))
    
    mean_vrfr = run_ensemble(prob, Tsit5(), jump)
    mean_vrdcb = run_ensemble(prob, Tsit5(), jump; vr_aggregator = VR_Direct())
    
    t = 10.0
    u0 = 0.2
    analytical_mean = u0 * exp(-t) + λ*(1 - exp(-t))

    @test isapprox(mean_vrfr, analytical_mean, rtol=0.05)
    @test isapprox(mean_vrfr, mean_vrdcb, rtol=0.05)
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
    birth_jump = VariableRateJump(birth_rate, birth_affect!)
    
    # Define death jump: X → ∅
    death_rate(u, p, t) = 0.5 * u[1]
    function death_affect!(integrator)
        integrator.u[1] -= 1
        integrator.p[3] += 1
        nothing 
    end
    death_jump = VariableRateJump(death_rate, death_affect!)

    Nsims = 100
    results = Dict()
    u0 = [1.0]
    tspan = (0.0, 10.0)
    for vr_aggregator in (VR_FRM(), VR_Direct())    
        jump_counts = zeros(Int, Nsims)
        p = [0.0, 0.0, 0]
        prob = ODEProblem(f, u0, tspan, p)
        jump_prob = JumpProblem(prob, Direct(), birth_jump, death_jump; vr_aggregator, rng)
        
        for i in 1:Nsims
            sol = solve(jump_prob, Tsit5())
            jump_counts[i] = jump_prob.prob.p[3]
            jump_prob.prob.p[3] = 0 
        end
        
        results[vr_aggregator] = (mean_jumps=mean(jump_counts), jump_counts=jump_counts)
        @test sum(jump_counts) > 1000
    end

    mean_jumps_vrfr = results[VR_FRM()].mean_jumps
    mean_jumps_vrdcb = results[VR_Direct()].mean_jumps
    @test isapprox(mean_jumps_vrfr, mean_jumps_vrdcb, rtol=0.1)
end
