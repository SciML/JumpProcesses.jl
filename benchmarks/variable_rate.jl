using DiffEqBase, JumpProcesses, OrdinaryDiffEq, StochasticDiffEq
using Random, LinearSolve
using StableRNGs

rng = StableRNG(12345)


# --- Test Case 1: Scalar ODE with Two Variable Rate Jumps ---
rate = (u, p, t) -> u[1]
affect! = (integrator) -> (integrator.u[1] = integrator.u[1] / 2)
jump = VariableRateJump(rate, affect!, interp_points = 1000)
jump2 = deepcopy(jump)

f = (du, u, p, t) -> (du[1] = u[1])
prob = ODEProblem(f, [0.2], (0.0, 10.0))
ensemble_prob = EnsembleProblem(prob)

jump_prob = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 1 Tsit5 - VRDirectCB: $time_gill, VRFRMODE: $time_next")

time_next = @elapsed solve(ensemble_prob, Rosenbrock23(autodiff = false), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, Rosenbrock23(autodiff = false), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 1 Rosenbrock23 (no autodiff) - VRDirectCB: $time_gill, VRFRMODE: $time_next")

time_next = @elapsed solve(ensemble_prob, Rosenbrock23(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, Rosenbrock23(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 1 Rosenbrock23 (autodiff) - VRDirectCB: $time_gill, VRFRMODE: $time_next")


# --- Test Case 2: Scalar SDE with Two Variable Rate Jumps ---
g = (du, u, p, t) -> (du[1] = u[1])
prob = SDEProblem(f, g, [0.2], (0.0, 10.0))
ensemble_prob = EnsembleProblem(prob)

jump_prob = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, SRIW1(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, SRIW1(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 2 SRIW1 - VRDirectCB: $time_gill, VRFRMODE: $time_next")


# --- Test Case 3: SDE with Parameter Switch ---
ff = (du, u, p, t) -> (du .= p == 0 ? 1.01u : 2.01u)
gg = (du, u, p, t) -> begin
    du[1, 1] = 0.3u[1]; du[1, 2] = 0.6u[1]
    du[2, 1] = 1.2u[1]; du[2, 2] = 0.2u[2]
end
rate_switch = (u, p, t) -> u[1] * 1.0
affect_switch! = (integrator) -> (integrator.p = 1)
jump_switch = VariableRateJump(rate_switch, affect_switch!)

prob = SDEProblem(ff, gg, ones(2), (0.0, 1.0), 0, noise_rate_prototype = zeros(2, 2))
ensemble_prob = EnsembleProblem(prob)

jump_prob = JumpProblem(prob, Direct(), jump_switch; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump_switch; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, SRA1(), EnsembleSerial(), dt = 1.0, trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, SRA1(), EnsembleSerial(), dt = 1.0, trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 3 SRA1 - VRDirectCB: $time_gill, VRFRMODE: $time_next")


# --- Test Case 4: ODE with Constant Rate Jump ---
f2 = (du, u, p, t) -> (du[1] = u[1])
prob = ODEProblem(f2, [0.2], (0.0, 10.0))
ensemble_prob = EnsembleProblem(prob)
rate2 = (u, p, t) -> 2
affect2! = (integrator) -> (integrator.u[1] = integrator.u[1] / 2)
jump = ConstantRateJump(rate2, affect2!)

jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 4 Tsit5 - VRDirectCB: $time_gill, VRFRMODE: $time_next")


# --- Test Case 5: ODE with Two Variable Rate Jumps (rate2b) ---
rate2b = (u, p, t) -> u[1]
jump = VariableRateJump(rate2b, affect2!)
jump2 = deepcopy(jump)

jump_prob = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 5 Tsit5 - VRDirectCB: $time_gill, VRFRMODE: $time_next")


# --- Test Case 6: SDE with Two Variable Rate Jumps (rate2b) ---
g2 = (du, u, p, t) -> (du[1] = u[1])
prob = SDEProblem(f2, g2, [0.2], (0.0, 10.0))
ensemble_prob = EnsembleProblem(prob)

jump_prob = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump, jump2; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, SRIW1(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, SRIW1(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 6 SRIW1 - VRDirectCB: $time_gill, VRFRMODE: $time_next")


# --- Test Case 7: Matrix ODE with Variable Rate Jump ---
f3 = (du, u, p, t) -> (du .= u)
prob = ODEProblem(f3, [1.0 2.0; 3.0 4.0], (0.0, 1.0))
ensemble_prob = EnsembleProblem(prob)
rate3 = (u, p, t) -> u[1] + u[2]
affect3! = (integrator) -> (integrator.u[1] = 0.25; integrator.u[2] = 0.5; integrator.u[3] = 0.75; integrator.u[4] = 1)
jump = VariableRateJump(rate3, affect3!)

jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 7 Tsit5 - VRDirectCB: $time_gill, VRFRMODE: $time_next")


# --- Test Case 8: Complex ODE with Variable Rate Jump ---
f4 = (dx, x, p, t) -> (dx[1] = x[1])
rate4 = (x, p, t) -> t
affect4! = (integrator) -> (integrator.u[1] = integrator.u[1] * 0.5)
jump = VariableRateJump(rate4, affect4!)
x₀ = 1.0 + 0.0im
prob = ODEProblem(f4, [x₀], (0.0, 6.0))
ensemble_prob = EnsembleProblem(prob)

jump_prob = JumpProblem(prob, Direct(), jump; vr_aggregator = VRFRMODE(), rng = rng)
jump_prob_gill = JumpProblem(prob, Direct(), jump; vr_aggregator = VRDirectCB(), rng = rng)

time_next = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob)
time_gill = @elapsed solve(ensemble_prob, Tsit5(), EnsembleSerial(), trajectories = 1000, jump_prob = jump_prob_gill)
println("Test 8 Tsit5 - VRDirectCB: $time_gill, VRFRMODE: $time_next")
