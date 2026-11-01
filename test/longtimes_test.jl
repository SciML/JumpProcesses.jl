using DiffEqBase, JumpProcesses, SciMLBase, Test, StableRNGs
using OrdinaryDiffEq, StochasticDiffEq, Statistics

rng = StableRNG(12345)

# test for https://github.com/SciML/JumpProcesses.jl/issues/177
p = [1.0, 2.0, 50.0]
ns = [[1 => 1], [1 => -1], [1 => 1]]
rs = [[1 => 1], [1 => 1], Pair{Int64, Int64}[]]
maj = MassActionJump(p, rs, ns)
u0 = [5]
tspan = (0.0, 2e6)
dt = tspan[2] / 1000
dprob = DiscreteProblem(u0, tspan, p)
jprob = JumpProblem(dprob, Direct(), maj, save_positions = (false, false))
sol = solve(jprob, SSAStepper(); saveat = tspan[1]:dt:tspan[2], rng)
@test length(unique(sol.u[(end - 10):end][:])) > 1

# Jump times that round to the current time in ODE and SDE problems. Each A, at rate 1, sets
# X = 1, and B, X → Y at rate 1e6 X, then fires within about 1e-6, which rounds to A's time
# for t ≥ 1e12. B's stop used to be at the current time, which the integrator never reaches,
# so no jump fired after the first A. 1e6 + 1 is exact, so a total rate updated
# incrementally keeps A's rate. A larger rate for B would also make `RDirect` slow, since it
# rejects candidates against the largest rate seen until it updates its maximum.
let
    rateA(u, p, t) = 1.0
    affectA!(integrator) = (integrator.u[1] += 1; nothing)
    rateB(u, p, t) = 1e6 * u[1]
    affectB!(integrator) = (integrator.u[1] -= 1; integrator.u[2] += 1; nothing)
    jumps = (ConstantRateJump(rateA, affectA!), ConstantRateJump(rateB, affectB!))
    dep_graph = [[1, 2], [2]]
    vartojumps_map = [[2], Int[]]
    jumptovars_map = [[1], [1, 2]]
    f!(du, u, p, t) = (du .= 0; nothing)
    g!(du, u, p, t) = (du .= 0; nothing)
    tspan = (1e12, 1e12 + 10)
    for (prob, alg) in ((ODEProblem(f!, [0.0, 0.0], tspan), Tsit5()),
            (SDEProblem(f!, g!, [0.0, 0.0], tspan), SRIW1())),
        aggregator in JumpProcesses.JUMP_AGGREGATORS

        jprob = JumpProblem(prob, aggregator, jumps...; dep_graph, vartojumps_map,
            jumptovars_map, save_positions = (false, false))
        rng = StableRNG(12345)
        us = [solve(jprob, alg; rng, save_everystep = false).u[end] for _ in 1:200]
        # every A was followed by its B, and A fired at rate 1 throughout
        @test all(u -> u[1] == 0, us)
        @test isapprox(mean(u -> u[2], us), 10; rtol = 0.1)
    end
end
