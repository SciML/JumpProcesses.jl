# prepares the problem
using JumpProcesses, Test, SymbolicIndexingInterface
rate1(u, p, t) = p[1]
rate2(u, p, t) = p[2]
affect1!(integ) = (integ.u[1] += 1)
affect2!(integ) = (integ.u[2] += 1)
crj1 = ConstantRateJump(rate1, affect1!)
crj2 = ConstantRateJump(rate2, affect2!)
maj = MassActionJump([[1 => 1], [1 => 1]], [[1 => -1], [1 => -1]]; param_idxs = [1,2])
g = DiscreteFunction((du, u, p, t) -> nothing;
    sys = SymbolicIndexingInterface.SymbolCache([:a, :b], [:p1, :p2], :t))
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0])
jprob = JumpProblem(dprob, Direct(), crj1, crj2, maj)

# test basic querying of u0 and p
@test jprob[:a] == 0
@test jprob[:b] == 10
@test getp(jprob,:p1)(jprob) == 1.0
@test getp(jprob,:p2)(jprob) == 2.0
@test jprob.ps[:p1] == 1.0
@test jprob.ps[:p2] == 2.0

# test updating u0
jprob[:a] = 20
@test jprob[:a] == 20

# test mass action jumps update with parameter mutation in problems
@test jprob.massaction_jump.scaled_rates[1] == 1.0
jprob.ps[:p1] = 3.0
@test jprob.ps[:p1] == 3.0
@test jprob.massaction_jump.scaled_rates[1] == 3.0
p1setter = setp(jprob, [:p1, :p2])
p1setter(jprob, [4.0, 10.0])
@test jprob.ps[:p1] == 4.0
@test jprob.ps[:p2] == 10.0
@test jprob.massaction_jump.scaled_rates == [4.0, 10.0]
