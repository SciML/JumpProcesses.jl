using DiffEqBase, DiffEqJump
using Test
const DJ = DiffEqJump

# test data
minpriority = 2.0^exponent(1e-12)
maxpriority = 2.0^exponent(1e12)
priorities = [1e-13, .99*minpriority, minpriority,1.01e-4, 1e-4, 5., 0., 1e10]


mingid = exponent(minpriority)   # = -40

# add two as 0. -> 1 and mingid -> minpriority
@inline function priortogid(priority, mingid)
    (priority <= eps(typeof(priority))) && return 1
    gid = exponent(priority) + 1
    (gid <= mingid) && return 2
    return gid - mingid + 2
end
ptog = priority -> priortogid(priority, mingid)

pt = DJ.PriorityTable(ptog, priorities, minpriority, maxpriority)

#display(priorities)
#display(pt)

# test insert
grpcnt = DJ.numgroups(pt)
DJ.insert!(pt, length(priorities)+1, maxpriority*.99)
@test grpcnt == DJ.numgroups(pt)
@test pt.groups[end].pids[1] == length(priorities)+1
DJ.insert!(pt, length(priorities)+2, maxpriority*.99999)
@test grpcnt == DJ.numgroups(pt)
@test pt.groups[end].pids[2] == length(priorities)+2

numsmall = length(pt.groups[2].pids)
DJ.insert!(pt, length(priorities)+3, minpriority*.6)
@test grpcnt == DJ.numgroups(pt)
@test pt.groups[2].pids[end] == length(priorities)+3

DJ.insert!(pt, length(priorities)+4, maxpriority)
@test grpcnt == DJ.numgroups(pt)-1
@test pt.groups[end].pids[1] == length(priorities)+4


# test updating
DJ.update!(pt, 5, priorities[5], 2*priorities[5])   # group 29
@test pt.groups[29].numpids == 1
@test pt.groups[30].numpids == 1

DJ.update!(pt, 9, maxpriority*.99, maxpriority*1.01)
@test pt.groups[end].numpids == 2
@test pt.groups[end-1].numpids == 1

DJ.update!(pt, 10, maxpriority*.99999, 0.)
@test pt.groups[1].numpids == 2


