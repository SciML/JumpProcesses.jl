using DiffEqBase, JumpProcesses
using Test
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
cnt = 0
Nsamps = Int(1e7)
for i in 1:Nsamps
    global cnt
    pid = DJ.sample(pt, priorities)
    (pid == 8) && (cnt += 1)
end
@test abs(cnt // Nsamps - 0.008968535978248484) / 0.008968535978248484 < 0.05
