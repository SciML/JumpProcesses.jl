"""
Dynamic table data structure to store and update priorities
Implementation
Stores ids that identify the priorities. Each 'group' stores a range of priority ids.
The basic design assumes a lower-most group of "zero" priorities, and a second
group storing all non-zero priorities that are < a `minpriority`. All other groups
store priorities within consecutive ranges. The `minpriority` is taken fixed at creation,
and the table will be padded with more groups dynamically based on inserted / updated priorities.
The ranges are assumed to be powers of two:
   bin 1 = {0},
   bin 2 = (0,`minpriority`),
   bin 3 = [`minpriority`,`2*minpriority`)...
   bin N = [`.5*maxpriority`,`maxpriority`)
"""

"return the index of the group where priority belongs"
function priortogid(priority, minpower)
    (priority <= eps(typeof(priority))) && return 1
    gid = exponent(priority) + 1
    (gid <= minpower) && return 2
    return gid - minpower + 2
end

mutable struct PriorityTable{F}

    minpower::Int # minimum power of 2

    "stores reaction ids for each group"
    gids::Vector{Vector{Int}}

    "stores number of reactions in each group"
    gnums::Vector{Int}

    "stores the sums of the priorities within each group"
    gsums::Vector{F}

    "stores the sum of the group sums"
    gsum::F

    "maps priority id to group and idx within the group"
    pidtogroup::Vector{Tuple{Int,Int}}
end

function PriorityTable(minpriority)
    F = typeof(minpriority)
    minpower = exponent(minpriority)
    gids = [Int[], Int[]] # initialize the first two groups
    gnums = [0,0]
    gsums = zeros(F, 2)
    gsum = zero(F)
    pidtogroup = Tuple{Int,Int}[]
    PriorityTable{F}(minpower, gids, gnums, gsums, gsum, pidtogroup)
end

function PriorityTable(minpriority, priorities)
    pt = PriorityTable(minpriority)
    for (pid, priority) in enumerate(priorities)
        insert!(pt, pid, priority)
    end
    pt
end

"sample a jump from pt"
function sample(pt::PriorityTable{F}, priorities, rng = rng=Random.GLOBAL_RNG) where F
    gsum = pt.gsum; gsums = pt.gsums

    # return id zero if total priority is zero
    (gsum < eps(F)) && return 0

    # sample group
    gid = 0
    r   = rand(rng) * gsum
    for i = length(gsums):-1:1
        r -= gsums[i]
        if r <= zero(r)
            gid = i
            break
        end        
    end

    # sample reaction within group
    pids = @view pt.gids[gid][1:pt.gnums[gid]]
    maxpriority = 2.0^(pt.minpower + gid-2)
    while true
        # pick a random element from pids[1:numpids]
        pid    = rand(rng, pids)
        # acceptance test
        ( rand(rng)*maxpriority < priorities[pid] ) && return pid
    end
end

"pad table to newgid"
function padtable!(pt::PriorityTable{F}, newgid) where F
    gids = pt.gids; gnums = pt.gnums; gsums = pt.gsums
    num_new_groups = newgid - length(gsums) # number of groups to add
    append!(gids, [Int[] for i in 1:num_new_groups])
    append!(gnums, zeros(Int, num_new_groups))
    append!(gsums, zeros(F, num_new_groups))
end

"insert pid in group gid"
function insert!(pt::PriorityTable, pid, gid::Int)
    # pad table if necessary
    if gid > length(pt.gnums)
        padtable!(pt, gid)
    end

    numpids = pt.gnums[gid]
    pids = pt.gids[gid]
    if numpids == length(pids)
        push!(pids, pid)
    else
        pids[numpids+1] = pid
    end
    # return the position of insertion
    pt.gnums[gid] = numpids+1
end

"update propensity of pid from oldpriority to newpriority"
function update!(pt::PriorityTable{F}, pid, oldpriority, newpriority) where F
    gsums = pt.gsums; pidtogroup = pt.pidtogroup; gnums = pt.gnums; gids = pt.gids

    oldgid, pididx = pidtogroup[pid]
    newgid = priortogid(newpriority, pt.minpower)

    # update the global priority sum
    pdiff    = newpriority - oldpriority
    pt.gsum += pdiff

    if oldgid == newgid
        # update the group priority too
        gsums[newgid] += pdiff
    else
        # remove pid from old group by moving the last pid to its place
        pids = gids[oldgid]
        lastpid = pids[gnums[oldgid]]
        pids[pididx] = lastpid
        gnums[oldgid] -= 1
    
        # update position in group of the pid that swapped places with the removed one
        pidtogroup[lastpid] = (oldgid,pididx)

        # insert the updated pid back and store its new position in the group
        pididx = insert!(pt, pid, newgid)
        pidtogroup[pid] = (newgid,pididx)

        # update sums, special case if group empty to avoid FP error in running sums
        gsums[oldgid]  = (gnums[oldgid] == 0) ? zero(F) : gsums[oldgid] - oldpriority
        gsums[newgid] += newpriority
    end
    nothing
end

"insert pid with priority in pt"
function insert!(pt::PriorityTable, pid, priority)
    # find group for new priority
    gid = priortogid(priority, pt.minpower)

    # update table and priority's group
    pididx = insert!(pt, pid, gid)
    pt.gsum += priority
    pt.gsums[gid] += priority

    if pid <= length(pt.pidtogroup)
        pt.pidtogroup[pid] = (gid,pididx)
    else
        push!(pt.pidtogroup, (gid,pididx))
    end
    nothing
end

function reset!(pt::PriorityTable{F}) where F
    pt.gnums .= 0
    pt.gsums .= zero(F)
    pt.gsum = 0
    fill!(pt.pidtogroup, (0,0))
end

function Base.show(io::IO, pt::PriorityTable)
    println(io, summary(pt))
    println(io, "minpower = ", pt.minpower)
    println(io, "sum of priorities = ", pt.gsum)
    println(io, "num of groups = ", length(pt.gsums))
    println(io, "pidtogroup = ", pt.pidtogroup)
    for (i,n) in enumerate(pt.gnums)
        if n > 0
            println(io, "group = $i\n  group sum = $(pt.gsums[i])\n  group num = $(pt.gnums[i])\n  group pids = $(pt.gids[i])")
        end
    end
end

function numgroups(pt::PriorityTable)
    @assert length(pt.gids) == length(pt.gnums) == length(pt.gsums)
    length(pt.gsums)
end

function numpriorities(pt::PriorityTable)
    length(pt.pidtogroup)
end

function groupsum(pt::PriorityTable)
    pt.gsum
end