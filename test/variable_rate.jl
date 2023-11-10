using DiffEqBase, JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Test
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
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)

integrator = init(jump_prob, Tsit5())

sol = solve(jump_prob, Tsit5())
sol = solve(jump_prob, Rosenbrock23(autodiff = false))
sol = solve(jump_prob, Rosenbrock23())

# @show sol[end]
# display(sol[end])

@test maximum([sol[i][2] for i in 1:length(sol)]) <= 1e-12
@test maximum([sol[i][3] for i in 1:length(sol)]) <= 1e-12

g = function (du, u, p, t)
    du[1] = u[1]
end

prob = SDEProblem(f, g, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)

sol = solve(jump_prob, SRIW1())

@test maximum([sol[i][2] for i in 1:length(sol)]) <= 1e-12
@test maximum([sol[i][3] for i in 1:length(sol)]) <= 1e-12

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
jump_prob = JumpProblem(prob, Direct(), jump_switch; rng = rng)
solve(jump_prob, SRA1(), dt = 1.0)

## Some integration tests

function f2(du, u, p, t)
    du[1] = u[1]
end

prob = ODEProblem(f2, [0.2], (0.0, 10.0))
rate2(u, p, t) = 2
affect2!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = ConstantRateJump(rate2, affect2!)
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
sol = solve(jump_prob, Tsit5())
sol(4.0)
sol[4]

rate2(u, p, t) = u[1]
affect2!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate2, affect2!)
jump2 = deepcopy(jump)
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)
sol = solve(jump_prob, Tsit5())
sol(4.0)
sol[4]

function g2(du, u, p, t)
    du[1] = u[1]
end

prob = SDEProblem(f2, g2, [0.2], (0.0, 10.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)
sol = solve(jump_prob, SRIW1())
sol(4.0)
sol[4]

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
jump_prob = JumpProblem(prob, Direct(), jump; rng = rng)
sol = solve(jump_prob, Tsit5())

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
jumpProblem = JumpProblem(prob, Direct(), jump)
sol = solve(jumpProblem, Tsit5())

# Out of place test

function drift(x, p, t)
    return p * x
end

function rate2(x, p, t)
    return 3 * max(0.0, x[1])
end

function affect!2(integrator)
    integrator.u ./= 2
end
x0 = rand(2)
prob = ODEProblem(drift, x0, (0.0, 10.0), 2.0)
jump = VariableRateJump(rate2, affect!2)
jump_prob = JumpProblem(prob, Direct(), jump)

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
