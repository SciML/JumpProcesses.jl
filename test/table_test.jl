# test PriorityTable
using DiffEqBase, DiffEqJump
const DJ = DiffEqJump
using Test

# test data
minpriority = 2.0^exponent(1e-12)
priorities = [1e-13, .99*minpriority, minpriority,1.01e-4, 1e-4, 5., 0., 1e10]

mingid = exponent(minpriority)   # = -40
pt = DJ.PriorityTable(minpriority, priorities)

# display(priorities)
# display(pt)

# test insert
grpcnt = DJ.numgroups(pt)
push!(priorities, priorities[end]*.99)
oldgnum = pt.gnums[end]
oldgsum = pt.gsums[end]
oldsum = pt.gsum
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt)
@test pt.pidtogroup[end][1] == grpcnt
@test pt.gnums[end] == oldgnum + 1
@test pt.gsums[end] ≈ oldgsum + priorities[end]
@test pt.gsum ≈ oldsum + priorities[end]

push!(priorities, priorities[end-1]*.99999)
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt)
@test pt.pidtogroup[end][1] == grpcnt

push!(priorities, minpriority*.6)
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt)
@test pt.pidtogroup[end][1] == 2

push!(priorities, priorities[end-2]*2)
DJ.insert!(pt, length(priorities), priorities[end])
@test grpcnt == DJ.numgroups(pt)-1
@test pt.pidtogroup[end][1] == DJ.numgroups(pt)

# test updating
DJ.update!(pt, 5, priorities[5], 2*priorities[5])   # group 29
priorities[5] *= 2
@test pt.gnums[29] == 1
@test pt.gnums[30] == 1

DJ.update!(pt, 9, priorities[9], priorities[9]*4)
priorities[9] = priorities[9]*4
@test pt.gnums[end] == 1
@test pt.gnums[end-1] == 1

DJ.update!(pt, 10, priorities[10], 0.)
priorities[10] = 0.
@test pt.gnums[1] == 2

# test sampling
Nsamps = Int(1e7)
for (j,prop) in enumerate(priorities)
    (prop/sum(priorities) <= 1/Nsamps) && continue
    cnt = 0
    for i = 1:Nsamps
        pid = DJ.sample(pt, priorities)
        (pid == j) && (cnt += 1)
    end
    @test abs(cnt/Nsamps - prop/sum(priorities)) / (prop/sum(priorities)) < 0.05
end

# test reset!
DJ.reset!(pt)
@test pt.gsum == 0.0
@test sum(pt.gnums) == 0
@test pt.pidtogroup[end] == (0,0)