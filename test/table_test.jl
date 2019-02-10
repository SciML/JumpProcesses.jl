using DiffEqBase, DiffEqJump, Test
const DJ = DiffEqJump

# test data
priorities = [1e-13, 1e-4, 5., 0., 1e10]
minpriority = 2^exponent(1e-12)
maxpriority = 2^exponent(1e12)


mingid = -exponent(minpriority)

# add two as 0. -> 1 and mingid -> minpriority
priortogid = priority -> priority <= eps(typeof(priority)) ? 1 : (exponent(priority) + mingid + 2)

pt = DJ.PriorityTable(priortogid, priorities, minpriority, maxpriority)

