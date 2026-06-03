# prepares the problem
using JumpProcesses, Test, SymbolicIndexingInterface
rate1(u, p, t) = p[1]
rate2(u, p, t) = p[2]
affect1!(integ) = (integ.u[1] += 1)
affect2!(integ) = (integ.u[2] += 1)
crj1 = ConstantRateJump(rate1, affect1!)
crj2 = ConstantRateJump(rate2, affect2!)
maj = MassActionJump([[1 => 1], [1 => 1]], [[1 => -1], [1 => -1]]; param_idxs = [1, 2])
g = DiscreteFunction((du, u, p, t) -> nothing;
    sys = SymbolicIndexingInterface.SymbolCache([:a, :b], [:p1, :p2], :t))
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0])
jprob = JumpProblem(dprob, Direct(), crj1, crj2, maj)

# test basic querying of u0 and p
@test jprob[:a] == 0
@test jprob[:b] == 10
@test getp(jprob, :p1)(jprob) == 1.0
@test getp(jprob, :p2)(jprob) == 2.0
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

# integrator tests
# note that `setu` is not currently supported as `set_u!` is not implemented for SSAStepper
integ = init(jprob, SSAStepper())
@test getu(integ, [:a, :b])(integ) == [20, 10]
integ[[:b, :a]] = [40, 5]
@test getu(integ, [:a, :b])(integ) == [5, 40]
@test getp(integ, :p2)(integ) == 10.0
setp(integ, :p2)(integ, 15.0)
@test getp(integ, :p2)(integ) == 15.0
@test jprob.massaction_jump.scaled_rates[2] == 10.0  # jump rate not updated
reset_aggregated_jumps!(integ)
@test jprob.massaction_jump.scaled_rates[2] == 15.0  # jump rate now updated

# remake tests
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0])
jprob = JumpProblem(dprob, Direct(), crj1, crj2, maj)
jprob = remake(jprob; u0 = [:a => -10, :b => 100], p = [:p2 => 3.5, :p1 => 0.5])
@test jprob.prob.u0 == [-10, 100]
@test jprob.prob.p == [0.5, 3.5]
@test jprob.massaction_jump.scaled_rates == [0.5, 3.5]
jprob = remake(jprob; u0 = [:b => 10], p = [:p2 => 4.5])
@test jprob.prob.u0 == [-10, 10]
@test jprob.prob.p == [0.5, 4.5]
@test jprob.massaction_jump.scaled_rates == [0.5, 4.5]

# test updating problems via regular indexing still updates the mass action jump
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0])
jprob = JumpProblem(dprob, Direct(), crj1, crj2, maj)
@test jprob.massaction_jump.scaled_rates[1] == 1.0
jprob.ps[1] = 3.0
@test jprob.ps[1] == 3.0
@test jprob.massaction_jump.scaled_rates[1] == 3.0

# test updating integrators via regular indexing
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0])
jprob = JumpProblem(dprob, Direct(), crj1, crj2, maj)
integ = init(jprob, SSAStepper())
integ.u .= [40, 5]
@test getu(integ, [1, 2])(integ) == [40, 5]
@test getp(integ, 2)(integ) == 2.0
@test integ.p[2] == 2.0
@test jprob.massaction_jump.scaled_rates[2] == 2.0
setp(integ, 2)(integ, 15.0)
@test integ.p[2] == 15.0
@test getp(integ, 2)(integ) == 15.0
reset_aggregated_jumps!(integ)
@test jprob.massaction_jump.scaled_rates[2] == 15.0  # jump rate now updated

# remake tests for regular indexing
dprob = DiscreteProblem(g, [0, 10], (0.0, 10.0), [1.0, 2.0])
jprob = JumpProblem(dprob, Direct(), crj1, crj2, maj)
jprob = remake(jprob; u0 = [-10, 100], p = [0.5, 3.5])
@test jprob.prob.u0 == [-10, 100]
@test jprob.prob.p == [0.5, 3.5]
@test jprob.massaction_jump.scaled_rates == [0.5, 3.5]
jprob = remake(jprob; u0 = [2 => 10], p = [2 => 4.5])
@test jprob.prob.u0 == [-10, 10]
@test jprob.prob.p == [0.5, 4.5]
@test jprob.massaction_jump.scaled_rates == [0.5, 4.5]
