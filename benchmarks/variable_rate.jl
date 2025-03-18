# This file is not directly included in a test case, but is used to
# benchmark and compare GillespieIntegCallback and NextReactionODE
using DiffEqBase, JumpProcesses, OrdinaryDiffEq, StochasticDiffEq, Test
using Random, LinearSolve
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> u[1]
affect! = (integrator) -> (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate, affect!, interp_points = 1000)
jump2 = deepcopy(jump)

f = function (du, u, p, t)
    du[1] = u[1]
end

prob = ODEProblem(f, [0.2], (0.0, 10.0))

jump_prob = JumpProblem(prob, Direct(), jump, jump2; variablerate_aggregator = NextReactionODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(),  jump, jump2; variablerate_aggregator = GillespieIntegCallback(), rng=rng)

integrator = init(jump_prob, Tsit5())
integrator = init(jump_prob_gill, Tsit5())

time_next = @elapsed solve(jump_prob, Tsit5())
time_gill = @elapsed solve(jump_prob_gill, Tsit5())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")


time_next = @elapsed solve(jump_prob,  Rosenbrock23(autodiff = false))
time_gill = @elapsed solve(jump_prob_gill,  Rosenbrock23(autodiff = false))

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")

time_next = @elapsed solve(jump_prob,  Rosenbrock23())
time_gill = @elapsed solve(jump_prob_gill,  Rosenbrock23())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")



g = function (du, u, p, t)
    du[1] = u[1]
end

prob = SDEProblem(f, g, [0.2], (0.0, 10.0))

jump_prob = JumpProblem(prob, Direct(), jump, jump2; variablerate_aggregator = NextReactionODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(),  jump, jump2; variablerate_aggregator = GillespieIntegCallback(), rng=rng)

time_next = @elapsed solve(jump_prob,  SRIW1())
time_gill = @elapsed solve(jump_prob_gill,  SRIW1())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")

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

jump_prob = JumpProblem(prob, Direct(), jump_switch; variablerate_aggregator = NextReactionODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump_switch; variablerate_aggregator = GillespieIntegCallback(), rng=rng)

time_next = @elapsed solve(jump_prob, SRA1(), dt = 1.0)
time_gill = @elapsed solve(jump_prob_gill, SRA1(), dt = 1.0)

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")



function f2(du, u, p, t)
    du[1] = u[1]
end

prob = ODEProblem(f2, [0.2], (0.0, 10.0))
rate2(u, p, t) = 2
affect2!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = ConstantRateJump(rate2, affect2!)

jump_prob = JumpProblem(prob, Direct(), jump; variablerate_aggregator = NextReactionODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump; variablerate_aggregator = GillespieIntegCallback(), rng=rng)

time_next = @elapsed solve(jump_prob, Tsit5())
time_gill = @elapsed solve(jump_prob_gill, Tsit5())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")


rate2b(u, p, t) = u[1]
affect2!(integrator) = (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate2b, affect2!)
jump2 = deepcopy(jump)

jump_prob = JumpProblem(prob, Direct(), jump, jump2; variablerate_aggregator = NextReactionODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump, jump2; variablerate_aggregator = GillespieIntegCallback(), rng=rng)

time_next = @elapsed solve(jump_prob, Tsit5())
time_gill = @elapsed solve(jump_prob_gill, Tsit5())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")




function g2(du, u, p, t)
    du[1] = u[1]
end

prob = SDEProblem(f2, g2, [0.2], (0.0, 10.0))

jump_prob = JumpProblem(prob, Direct(), jump, jump2; variablerate_aggregator = NextReactionODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump, jump2; variablerate_aggregator = GillespieIntegCallback(), rng=rng)

time_next = @elapsed solve(jump_prob, SRIW1())
time_gill = @elapsed solve(jump_prob_gill, SRIW1())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")




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

jump_prob = JumpProblem(prob, Direct(), jump; variablerate_aggregator = NextReactionODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump; variablerate_aggregator = GillespieIntegCallback(), rng=rng)

time_next = @elapsed solve(jump_prob, Tsit5())
time_gill = @elapsed solve(jump_prob_gill, Tsit5())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")




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

jump_prob = JumpProblem(prob, Direct(), jump; variablerate_aggregator = NextReactionODE())
jump_prob_gill = JumpProblem(prob, Direct(), jump; variablerate_aggregator = GillespieIntegCallback())

time_next = @elapsed solve(jump_prob, Tsit5())
time_gill = @elapsed solve(jump_prob_gill, Tsit5())

println("Time taken for GillespieIntegCallback $time_gill")
println("Time taken for NextReactionODE $time_next")