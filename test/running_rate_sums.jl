using JumpProcesses, OrdinaryDiffEq, SciMLBase, StableRNGs, Test
const JP = JumpProcesses

# The aggregators that keep running sums of their rates (or rate bounds), the sum, its bound on
# rounding errors, and the rates it sums
const RUNNING_SUM_AGGREGATORS = (SortingDirect(), RDirect(), DirectCR(), RSSA(), RSSACR())
running_sum(a::JP.DirectCRJumpAggregation) = (a.rt.gsum, a.sum_rate_err, a.cur_rates)
running_sum(a::JP.RSSACRJumpAggregation) = (a.rt.gsum, a.sum_rate_err, a.cur_rate_high)
running_sum(a::JP.RSSAJumpAggregation) = (a.sum_rate, a.sum_rate_err, a.cur_rate_high)
running_sum(a) = (a.sum_rate, a.sum_rate_err, a.cur_rates)

sir = MassActionJump([[1 => 1, 2 => 1], [2 => 1]], [[1 => -1, 2 => 1], [2 => -1, 3 => 1]];
    param_idxs = [1, 2])

# Once the epidemic ends every rate is zero, but rounding used to leave the running sums
# slightly positive or negative. With a positive sum RDirect's rejection loop never ended
# (seed 3), and before a long enough end time SortingDirect ran jumps with zero rate and
# DirectCR sampled an invalid jump.
@testset "Running rate sums are zero after extinction" begin
    dprob = DiscreteProblem([999, 50, 0], (0.0, 1e30), (0.1, 100.0))
    for agg in RUNNING_SUM_AGGREGATORS
        jprob = JumpProblem(dprob, agg, sir; save_positions = (false, false))
        sols = [solve(jprob, SSAStepper(); rng = StableRNG(seed)) for seed in 1:50]
        @test all(sol -> sol.u[end][2] == 0 && all(>=(0), sol.u[end]), sols)
        @test running_sum(jprob.discrete_jump_aggregation)[1] == 0
    end
end

# A negative sum gave a next jump time far in the past, so `add_tstop!` threw in ODE solves
# (seed 18 for DirectCR, 19 for SortingDirect)
@testset "No jump times in the past after extinction in ODE solves" begin
    f!(du, u, p, t) = (du .= 0; du[4] = 0.01 * u[2]; nothing)
    oprob = ODEProblem(f!, [999.0, 1.0, 0.0, 0.0], (0.0, 250.0), (1e-4, 0.01))
    for agg in RUNNING_SUM_AGGREGATORS
        jprob = JumpProblem(oprob, agg, sir; save_positions = (false, false))
        sols = [solve(jprob, Tsit5(); rng = StableRNG(seed), saveat = 10.0)
                for seed in 1:20]
        @test all(sol -> SciMLBase.successful_retcode(sol), sols)
    end
end

# After every jump, the running sum is within its error bound of the exact sum of the rates
@testset "Running rate sum error bounds hold" begin
    # A1 <-> A2 <-> ... <-> A6 with rate constants from 1e-3 to 10, births of A1 and deaths
    # of A6
    k = [10.0^(i - 4) for i in 1:5]
    substoich = [[[i => 1] for i in 1:5]; [[i + 1 => 1] for i in 1:5]; [Pair{Int, Int}[]];
                 [[6 => 1]]]
    netstoich = [[[i => -1, i + 1 => 1] for i in 1:5]; [[i => 1, i + 1 => -1] for i in 1:5];
                 [[1 => 1]]; [[6 => -1]]]
    chain = MassActionJump([k; reverse(k); 50.0; 1.0], substoich, netstoich)
    cases = ((DiscreteProblem([999, 50, 0], (0.0, 1e3), (0.1, 100.0)), sir),
        (DiscreteProblem(fill(100, 6), (0.0, 5.0)), chain))
    for (prob, jumps) in cases
        for agg in RUNNING_SUM_AGGREGATORS
            jprob = JumpProblem(prob, agg, jumps; save_positions = (false, false))
            a = jprob.discrete_jump_aggregation
            ok = Ref(true)
            function check!(integrator)
                s, bound, rs = running_sum(a)
                ok[] &= abs(big(s) - sum(big, rs; init = big(0.0))) <= bound
                u_modified!(integrator, false)
            end
            cb = DiscreteCallback((u, t, integrator) -> true, check!;
                save_positions = (false, false))
            for seed in 1:5
                solve(jprob, SSAStepper(); rng = StableRNG(seed), callback = cb)
            end
            @test ok[]
        end
    end
end

# Four fast species die out within about 1e-8, then B decays slowly, so
# B(t) ~ Binomial(10, exp(-t / 1000)). Rounding errors from the fast species' large rates used
# to build up in the running sums and stay there, so B decayed too slowly. RDirect is left
# out: it rejects candidates against the largest rate seen until it updates its maximum, which
# takes very long after such a drop.
@testset "Accurate rate sums after a population crash" begin
    maj = MassActionJump([fill(1e9, 4); 1e-3], [[i => 1] for i in 1:5],
        [[i => -1] for i in 1:5])
    ts = [500.0, 1000.0, 2000.0]
    dprob = DiscreteProblem([fill(2000, 4); 10], (0.0, 2000.0))
    p = exp.(-ts ./ 1000)
    N = 4000
    for agg in (SortingDirect(), DirectCR(), RSSA(), RSSACR())
        jprob = JumpProblem(dprob, agg, maj; save_positions = (false, false))
        means = zeros(length(ts))
        for seed in 1:N
            sol = solve(jprob, SSAStepper(); rng = StableRNG(seed), saveat = ts)
            means .+= [sol(t)[5] for t in ts] ./ N
        end
        z = (means .- 10 .* p) ./ sqrt.(10 .* p .* (1 .- p) ./ N)
        @test all(abs.(z) .< 4.5)
    end
end

# Recomputing a priority table's sums gives the exact group sums and total
@testset "PriorityTable resum!" begin
    priorities = [1e-13, 1e-4, 5.0, 0.0, 1e10, 3.0, 1e10 + 1]
    minpriority = 2.0^exponent(1e-12)
    ptog = priority -> JP.priortogid(priority, exponent(minpriority))
    pt = JP.PriorityTable(ptog, priorities, minpriority, 2.0^exponent(1e12))
    for (pid, newpriority) in ((5, 0.0), (7, 2.0), (3, 1e11), (5, 1e-5), (3, 0.0))
        JP.update!(pt, pid, priorities[pid], newpriority)
        priorities[pid] = newpriority
    end
    JP.resum!(pt, priorities)
    @test pt.gsum == sum(pt.gsums)
    @test all(pt.gsums[gid] == sum(priorities[group.pids[1:(group.numpids)]]; init = 0.0)
    for (gid, group) in enumerate(pt.groups))
end
