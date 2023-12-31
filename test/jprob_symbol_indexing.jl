# prepares the problem
using JumpProcesses, Test
rate1(u, p, t) = p[1]
rate2(u, p, t) = p[2]
affect1!(integ) = (integ.u[1] += 1)
affect2!(integ) = (integ.u[2] += 1)
crj1 = ConstantRateJump(rate1, affect1!)
crj2 = ConstantRateJump(rate2, affect2!)
g = DiscreteFunction((du, u, p, t) -> nothing; sys = SymbolicIndexingInterface.SymbolCache([:a, :b], [:p1, :p2], :t)
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0])
jprob = JumpProblem(dprob, Direct(), crj1, crj2)

# runs the tests
@test jprob[:a] == 0
@test jprob[:b] == 10

# these are no longer supported by SciMLBase
@test getp(jprob,:p1)(jprob) == 1.0
@test getp(jprob,:p2)(jprob) == 2.0
