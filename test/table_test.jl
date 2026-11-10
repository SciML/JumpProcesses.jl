using DiffEqBase, JumpProcesses
using Test
using StableRNGs
const DJ = JumpProcesses

# test data
minpriority = 2.0^exponent(1e-12)
maxpriority = 2.0^exponent(1e12)
priorities = [1e-13, 0.99 * minpriority, minpriority, 1.01e-4, 1e-4, 5.0, 0.0, 1e10]

mingid = exponent(minpriority)   # = -40
ptog = priority -> DJ.priortogid(priority, mingid)
pt = DJ.PriorityTable(ptog, priorities, minpriority, maxpriority)

#display(priorities)
#display(pt)

# test insert
grpcnt = DJ.numgroups(pt)
push!(priorities, maxpriority * 0.99)
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt)
@test pt.groups[end].pids[1] == length(priorities)

push!(priorities, maxpriority * 0.99999)
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt)
@test pt.groups[end].pids[2] == length(priorities)

numsmall = length(pt.groups[2].pids)
push!(priorities, minpriority * 0.6)
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt)
@test pt.groups[2].pids[end] == length(priorities)

push!(priorities, maxpriority)
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt) - 1
@test pt.groups[end].pids[1] == length(priorities)

# test updating
DJ.update!(pt, 5, priorities[5], 2 * priorities[5])   # group 29
priorities[5] *= 2
@test pt.groups[29].numpids == 1
@test pt.groups[30].numpids == 1

DJ.update!(pt, 9, priorities[9], maxpriority * 1.01)
priorities[9] = maxpriority * 1.01
@test pt.groups[end].numpids == 2
@test pt.groups[end - 1].numpids == 1

DJ.update!(pt, 10, priorities[10], 0.0)
priorities[10] = 0.0
@test pt.groups[1].numpids == 2

# test sampling
rng = StableRNG(12345)
cnt = 0
Nsamps = Int(1e7)
for i in 1:Nsamps
    global cnt
    pid = DJ.sample(pt, priorities, rng)
    (pid == 8) && (cnt += 1)
end
@test abs(cnt // Nsamps - 0.008968535978248484) / 0.008968535978248484 < 0.05

##### PRIORITY TIME TABLE TESTS FOR CCNRM
mintime = 0.0;
maxtime = 100.0;
timestep = 1.5 / 16;
times = [2.0, 8.0, 13.0, 15.0, 74.0]

ptt = DJ.PriorityTimeTable(times, mintime, timestep)
@test DJ.getfirst(ptt) == (1, 2.0)
@test ptt.pidtogroup[5] == (0, 0) # Should not store the last one, outside the window.

# Test update
DJ.update!(ptt, 1, times[1], 10 * times[1]) # 2. -> 20., group 2 to group 14
@test ptt.groups[14].numpids == 1
@test DJ.getfirst(ptt) == (2, 8.0)
# Updating beyond the time window should not change the max priority. 
DJ.update!(ptt, 1, times[1], 70.0) # 20. -> 70.
@test ptt.groups[14].numpids == 0
@test ptt.maxtime == 66.0
@test ptt.pidtogroup[1] == (0, 0)

# Test rebuild
for i in 1:4
    DJ.update!(ptt, i, times[i], times[i] + 66.0)
end
@test DJ.getfirst(ptt) === (0, 0) # No more left.

mintime = 66.0;
timestep = 0.75 / 16;
DJ.rebuild!(ptt, mintime, timestep)
@test ptt.groups[11].numpids == 2 # 73.5-74.25
@test ptt.groups[18].numpids == 1
@test ptt.groups[21].numpids == 1
@test ptt.pidtogroup[1] == (0, 0)

##### Float32 tests (issue #674)
minpriority32 = Float32(2.0)^exponent(Float32(1e-12))
maxpriority32 = Float32(2.0)^exponent(Float32(1e12))
priorities32 = [Float32(1e-13), 0.99f0 * minpriority32, minpriority32, Float32(1.01e-4),
    Float32(1e-4), 5.0f0, 0.0f0, Float32(1e10)]
mingid32 = exponent(minpriority32)
ptog32 = priority -> DJ.priortogid(priority, mingid32)
pt32 = DJ.PriorityTable(ptog32, priorities32, minpriority32, maxpriority32)
@test eltype(priorities32) == Float32
@test typeof(pt32.minpriority) == Float32
@test typeof(pt32.maxpriority) == Float32
@test eltype(pt32.groups[1].maxpriority) == Float32

# insert a priority far above the initial maxpriority (table must grow, not hang)
push!(priorities32, Float32(1e20))
DJ.insert!(pt32, length(priorities32), priorities32[end])
@test pt32.groups[end].pids[end] == length(priorities32)

# update across groups
DJ.update!(pt32, 5, priorities32[5], 8.0f0)
priorities32[5] = 8.0f0

# sample returns valid ids and groupsum tracks the priorities
rng32 = StableRNG(12345)
samp = DJ.sample(pt32, priorities32, rng32)
@test 1 <= samp <= length(priorities32)
DJ.resum!(pt32, priorities32)
@test isapprox(DJ.groupsum(pt32), sum(Float64.(priorities32)); rtol = 1e-6)

# table built the way DirectCR builds it (maxpriority = 2*minpriority below
# eps(Float32)): this hung pre-fix, as the constructor collapsed to zero groups and
# zero maxpriority, spinning `padtable!` forever on `0 * 2`
crmin = Float32(2)^exponent(Float32(1e-12))
crtog = priority -> DJ.priortogid(priority, exponent(crmin))
crpt = DJ.PriorityTable(crtog, zeros(Float32, 1), crmin, 2 * crmin)
@test DJ.numgroups(crpt) == 3
@test typeof(crpt.maxpriority) == Float32
@test crpt.maxpriority > 0
DJ.insert!(crpt, 1, 10.0f0)
@test DJ.groupsum(crpt) == 10.0f0
DJ.insert!(crpt, 2, 0.0f0)
@test DJ.sample(crpt, Float32[10.0f0, 0.0f0], rng32) == 1

# PriorityTimeTable with Float32 times; update with a wider (Float64) time, as
# `t + randexp(rng) / rate` produces in CCNRM
times32 = Float32[2.0, 8.0, 13.0, 15.0, 74.0]
ptt32 = DJ.PriorityTimeTable(times32, 0.0f0, 1.5f0 / 16)
@test DJ.getfirst(ptt32) == (1, 2.0f0)
DJ.update!(ptt32, 1, times32[1], 20.0)  # Float64 newtime
@test ptt32.times[1] == 20.0f0
@test DJ.getfirst(ptt32) == (2, 8.0f0)
DJ.update!(ptt32, 1, 20.0f0, 70.0)  # Float64 newtime outside the window
@test ptt32.pidtogroup[1] == (0, 0)

# full solves with Float32 rates and tspan (issue #674 reproducer)
majump32 = MassActionJump([10.0f0, 1.0f0], [Pair{Int, Int}[], [1 => 1]], [
    [1 => 1], [1 => -1]])
dprob32 = DiscreteProblem([0], (0.0f0, 10.0f0))
for agg in (Direct(), DirectCR(), RSSACR(), CCNRM())
    sol = solve(JumpProblem(dprob32, agg, majump32), SSAStepper())
    @test eltype(sol.t) == Float32
    @test sol.u[end][1] >= 0
end

# BigFloat tspan works for the priority-table aggregators (issue #674)
majumpbig = MassActionJump([10.0, 1.0], [Pair{Int, Int}[], [1 => 1]], [[1 => 1], [1 => -1]])
dprobbig = DiscreteProblem([0], (BigFloat(0), BigFloat(10)))
for agg in (DirectCR(), RSSACR(), CCNRM())
    sol = solve(JumpProblem(dprobbig, agg, majumpbig), SSAStepper())
    @test sol.u[end][1] >= 0
    @test eltype(sol.t) == BigFloat
end
