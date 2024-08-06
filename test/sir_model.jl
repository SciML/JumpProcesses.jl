using JumpProcesses, DiffEqBase, OrdinaryDiffEq
using Test
using StableRNGs
rng = StableRNG(12345)

rate = (u, p, t) -> (0.1 / 1000.0) * u[1] * u[2]
affect! = function (integrator)
    integrator.u[1] -= 1
    integrator.u[2] += 1
end
jump = ConstantRateJump(rate, affect!)

rate = (u, p, t) -> 0.01u[2]
affect! = function (integrator)
    integrator.u[2] -= 1
    integrator.u[3] += 1
end
jump2 = ConstantRateJump(rate, affect!)

prob = DiscreteProblem([999.0, 1.0, 0.0], (0.0, 250.0))
jump_prob = JumpProblem(prob, Direct(), jump, jump2; rng = rng)
integrator = init(jump_prob, FunctionMap())

condition(u, t, integrator) = t == 100
function purge_affect!(integrator)
    integrator.u[2] ÷= 10
    reset_aggregated_jumps!(integrator)
end
cb = DiscreteCallback(condition, purge_affect!, save_positions = (false, false))
sol = solve(jump_prob, FunctionMap(), callback = cb, tstops = [100])
sol = solve(jump_prob, SSAStepper(), callback = cb, tstops = [100])

# test README example using the auto-solver selection runs
let
    # here we order S = 1, I = 2, and R = 3
    # substrate stoichiometry:
    substoich = [[1 => 1, 2 => 1],    # 1*S + 1*I
    [2 => 1]]                     # 1*I
    # net change by each jump type
    netstoich = [[1 => -1, 2 => 1],   # S -> S-1, I -> I+1
    [2 => -1, 3 => 1]]            # I -> I-1, R -> R+1
    # rate constants for each jump
    p = (0.1 / 1000, 0.01)

    # p[1] is rate for S+I --> 2I, p[2] for I --> R
    pidxs = [1, 2]

    maj = MassActionJump(substoich, netstoich; param_idxs = pidxs)

    u₀ = [999, 1, 0]       #[S(0),I(0),R(0)]
    tspan = (0.0, 250.0)
    dprob = DiscreteProblem(u₀, tspan, p)

    # use the Direct method to simulate
    jprob = JumpProblem(dprob, maj)

    # solve as a pure jump process, i.e. using SSAStepper
    sol = solve(jprob)
end