#using DiffEqBase, DiffEqJump, Test
#const DJ = DiffEqJump

# test data
priorities = [1e-13, 1e-4, 5., 0., 1e10]
minpriority = 2.0^exponent(1e-12)
maxpriority = 2.0^exponent(1e12)


mingid = -exponent(minpriority)

# add two as 0. -> 1 and mingid -> minpriority
@inline function priortogid2(priority, mingid)
    (priority <= eps(typeof(priority))) && return 1
    gid = exponent(priority)
    (gid <= mingid) && return 2
    return gid + mingid + 2
end
ptog = priority -> priortogid2(priority, mingid)

pt = PriorityTable(ptog, priorities, minpriority, maxpriority)

