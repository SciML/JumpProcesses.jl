using JumpProcesses, OrdinaryDiffEq, Test
using StableRNGs
rng = StableRNG(12345)

# test that we only save when a jump occurs
for alg in (Coevolve(),)
    u0 = [0]
    tspan = (0.0, 30.0)

    dprob = DiscreteProblem(u0, tspan)
    # set the rate to 0, so that no jump ever occurs; but urate is positive so
    # Coevolve will consider many candidates before the end of the simmulation.
    # None of these points should be saved.
    jump = VariableRateJump((u, p, t) -> 0, (integrator) -> integrator.u[1] += 1;
                            urate = (u, p, t) -> 1.0, rateinterval = (u, p, t) -> 5.0)
    jumpproblem = JumpProblem(dprob, alg, jump; dep_graph = [[1]],
                              save_positions = (false, true))
    sol = solve(jumpproblem, SSAStepper())
    @test sol.t == [0.0, 30.0]

    oprob = ODEProblem((du, u, p, t) -> 0, u0, tspan)
    jump = VariableRateJump((u, p, t) -> 0, (integrator) -> integrator.u[1] += 1;
                            urate = (u, p, t) -> 1.0, rateinterval = (u, p, t) -> 5.0)
    jumpproblem = JumpProblem(oprob, alg, jump; dep_graph = [[1]],
                              save_positions = (false, true))
    sol = solve(jumpproblem, Tsit5(); save_everystep = false)
    @test sol.t == [0.0, 30.0]
end
