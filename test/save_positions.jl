using JumpProcesses, OrdinaryDiffEq, Test, SciMLBase
using StableRNGs
rng = StableRNG(12345)

# test that we only save when a jump occurs
let
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
            save_positions = (false, true), rng)
        sol = solve(jumpproblem, SSAStepper())
        @test sol.t == [0.0, 30.0]

        oprob = ODEProblem((du, u, p, t) -> 0, u0, tspan)
        jump = VariableRateJump((u, p, t) -> 0, (integrator) -> integrator.u[1] += 1;
            urate = (u, p, t) -> 1.0, rateinterval = (u, p, t) -> 5.0)
        jumpproblem = JumpProblem(oprob, alg, jump; dep_graph = [[1]],
            save_positions = (false, true), rng)
        sol = solve(jumpproblem, Tsit5(); save_everystep = false)
        @test sol.t == [0.0, 30.0]
    end
end

# test isdenseplot gives correct values for SSAStepper and non-SSAStepper models
let
    rate(u, p, t) = max(u[1],0.0)
    affect!(integ) = (integ.u[1] -= 1; nothing)
    crj = ConstantRateJump(rate, affect!)
    u0 = [10.0]
    tspan = (0.0, 10.0)
    dprob = DiscreteProblem(u0, tspan)
    sps = ((true, true), (true, false), (false, true), (false, false))

    # for pure jump problems dense = save_everystep
    vals = (true, true, true, false)
    for (sp,val) in zip(sps, vals)
        jprob = JumpProblem(dprob, Direct(), crj; save_positions = sp, rng)
        sol = solve(jprob, SSAStepper())
        @test SciMLBase.isdenseplot(sol) == val
    end

    # for mixed problems sol.dense currently ignores save_positions
    oprob = ODEProblem((du,u,p,t) -> du[1] = .1, u0, tspan)
    for sp in sps
        jprob = JumpProblem(oprob, Direct(), crj; save_positions = sp, rng)
        sol = solve(jprob, Tsit5())
        @test sol.dense == true
        @test SciMLBase.isdenseplot(sol) == true
    end

end