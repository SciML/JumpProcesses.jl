# prepares the problem
using JumpProcesses, Test
rate1(u, p, t) = p[1]
rate2(u, p, t) = p[2]
affect1!(integ) = (integ.u[1] += 1)
affect2!(integ) = (integ.u[2] += 1)
crj1 = ConstantRateJump(rate1, affect1!)
crj2 = ConstantRateJump(rate2, affect2!)
g = DiscreteFunction((du, u, p, t) -> nothing; syms = [:a, :b], paramsyms = [:p1, :p2])
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0]; syms = [:a, :b],
                        paramsyms = [:p1, :p2])
jprob = JumpProblem(dprob, Direct(), crj1, crj2)

# runs the tests
@test jprob[:a] == 0
@test jprob[:b] == 10
@test jprob[:p1] == 1.0
@test jprob[:p2] == 2.0
