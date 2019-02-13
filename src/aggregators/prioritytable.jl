"""
Dynamic table data structure to store and update priorities

Implementation
Stores ids that identify the priorities. Each group stores a range of priority ids. 
The basic design assumes a lower-most group of "zero" priorities, and a second
group storing all non-zero priorities that are < a `minpriority`. All other groups
store priorities within consecutive ranges. The `minpriority` is taken fixed at creation,
but the `maxpriority` will increase dynamically based on inserted / updated priorities.
The ranges are assumed to be powers of two:
   bin 1 = {0}, 
   bin 2 = (0,`minpriority`), 
   bin 3 = [`minpriority`,`2*minpriority`)...
   bin N = [`.5*maxpriority`,`maxpriority`)
*Assumes* the `priortogid` function that maps priorities to groups maps the upper end of 
the interval to the next group. i.e. maxpriority -> N+1
"""

"""
One group (i.e. bin) of priority ids within the table
"""
mutable struct PriorityGroup{T,W <: AbstractVector}
    "(strict) upper bound for priorities in this group"
    maxpriority::T

    "number of priority ids"
    numpids::Int

    "priority ids associated with group"
    pids::W
end
PriorityGroup{U}(maxpriority::T) where {T,U} = PriorityGroup(maxpriority, 0, Vector{U}())

@inline function insert!(pg::PriorityGroup, pid)
    @unpack numpids, pids = pg

    if numpids == length(pids)
        push!(pids, pid)
    else
        @inbounds pids[numpids+1] = pid
    end

    # return the position of insertion
    pg.numpids += 1
end

@inline function remove!(pg::PriorityGroup, pididx)
    @unpack numpids, pids = pg

    @inbounds lastpid = pids[numpids]

    # simply swap the last id with the one to remove
    @inbounds pids[pididx] = lastpid    
    pg.numpids -= 1

    # return the pid that was swapped to pididx
    lastpid
end

function Base.show(io::IO, pg::PriorityGroup)
    println("  ", summary(pg))
    println("  maxpriority = ", pg.maxpriority)
    println("  numpids = ", pg.numpids)
    println("  pids = ", pg.pids[1:pg.numpids])
end


"""
Table to store the groups.
"""
mutable struct PriorityTable{F,S,T,U<:Function}
    "non-zero values below this are binned together, static"
    minpriority::F   

    "values above this cause new groups to be added"
    maxpriority::F   

    "bins storing priority ids within a given range"
    groups::Vector{PriorityGroup{F,Vector{S}}}

    "stores the sums of the priorities within each group"
    gsums::Vector{F}

    "stores the sum of the group sums"
    gsum::F

    "maps priority id to group and idx within the group"
    pidtogroup::Vector{Tuple{T,T}}

    "mapping from priority value to group id that stores it"
    priortogid::U
end

"""
Setup table from a vector of priorities. The id
of a priority is its position within this vector.
"""
function PriorityTable(priortogid::Function, priorities::AbstractVector, minpriority, maxpriority)

    numgroups  = priortogid(maxpriority)
    numgroups -= one(typeof(numgroups))
    pidtype    = typeof(numgroups)
    ptype      = eltype(priorities)
    groups     = Vector{PriorityGroup{ptype,Vector{pidtype}}}()
    pidtogroup = Vector{Tuple{Int,Int}}(undef, length(priorities))
    gsum       = zero(ptype)
    gsums      = zeros(ptype, numgroups)

    # create the groups, {0}, (0,minpriority), [minpriority,2*minpriority)...
    push!(groups, PriorityGroup{pidtype}(zero(ptype)))
    gmaxprior = minpriority
    for i = 2:numgroups
        push!(groups, PriorityGroup{pidtype}(gmaxprior))
        gmaxprior *= 2
    end
    #@assert abs(gmaxprior - 2*maxpriority) <= eps(typeof(maxpriority)) "gmaxprior = $gmaxprior, maxpriority=$maxpriority"

    pt = PriorityTable(minpriority, maxpriority, groups, gsums, gsum, pidtogroup, priortogid)

    # insert priority ids into the groups
    for (pid,priority) in enumerate(priorities)
        insert!(pt, pid, priority)
    end

    pt
end

########################## ACCESSORS ##########################
@inline function numgroups(pt::PriorityTable)
    length(pt.groups)
end

@inline function numpriorities(pt::PriorityTable)
    length(pt.pidtogroup)
end

@inline function groupsum(pt::PriorityTable)
    pt.gsum
end

"""
Adds extra groups to the table to accomodate a new maxpriority.
"""
function padtable!(pt::PriorityTable, pid, priority)
    @unpack maxpriority, groups, gsums = pt
    pidtype = typeof(pid)

    while priority >= maxpriority
        maxpriority *= 2
        push!(groups, PriorityGroup{pidtype}(maxpriority))
        push!(gsums, zero(eltype(gsums)))
    end
    pt.maxpriority = maxpriority
    nothing
end

# assumes pid is at most 1 greater than last priority (id) currently in table
# i.e. pid = length(pidtogroup) + 1
function insert!(pt::PriorityTable, pid, priority)  
    @unpack maxpriority, groups, gsums, pidtogroup, priortogid = pt
    pidtype = typeof(pid)

    # find group for new priority
    gid = priortogid(priority)

    # add new (empty) groups if priority is too big
    if priority >= maxpriority
        padtable!(pt, pid, priority)
        #@assert (gid == length(groups))
    end

    # update table and priority's group
    pt.gsum += priority
    @inbounds gsums[gid] += priority
    @inbounds pididx = insert!(groups[gid], pid)

    if pid <= length(pidtogroup)
        @inbounds pidtogroup[pid] = (gid,pididx)
    else
        #@assert pid == length(pidtogroup) + 1
        push!(pidtogroup, (gid,pididx))
    end

    nothing
end

function update!(pt::PriorityTable, pid, oldpriority, newpriority)
    @unpack maxpriority, groups, gsums, pidtogroup, priortogid = pt
    
    oldgid = priortogid(oldpriority)
    newgid = priortogid(newpriority)    

    # expand the table if necessary
    if newpriority >= maxpriority
        padtable!(pt, pid, newpriority)
    end

    # update the global priority sum
    pdiff    = newpriority - oldpriority
    pt.gsum += pdiff

    if oldgid == newgid
        # update the group priority too
        @inbounds gsums[newgid] += pdiff
    else        
        @inbounds begin
            # location in oldgid of pid to move
            pididx = pidtogroup[pid][2]

            # remove pid from old group
            movedpid = remove!(groups[oldgid], pididx)

            # update position in group of the pid that swapped places with the removed one
            pidtogroup[movedpid] = (oldgid,pididx)

            # insert the updated pid back and store it's new position in the group
            pididx = insert!(groups[newgid], pid)
            pidtogroup[pid] = (newgid,pididx)

            # update sums, special case if group empty to avoid FP error in running sums
            grpsz = groups[oldgid].numpids
            gsums[oldgid]  = (grpsz == zero(grpsz)) ? zero(oldpriority) : gsums[oldgid] - oldpriority
            gsums[newgid] += newpriority
        end
    end
    nothing
end

function Base.show(io::IO, pt::PriorityTable)
    println(summary(pt))
    println("(minpriority,maxpriority) = ", (pt.minpriority, pt.maxpriority))    
    println("sum of priorities = ", pt.gsum)
    println("num of groups = ", length(pt.groups))
    println("pidtogroup = ", pt.pidtogroup)
    for (i,group) in enumerate(pt.groups)
        if group.numpids > 0
            println("group = ",i,", group sum = ", pt.gsums[i])
            Base.show(io,group)
        end
    end
end

#######################################################
# routines for DirectCR
#######################################################

# map priority (i.e. jump rate) to integer
# add two as 0. -> 1 and  priority < minpriority ==> pid -> 2
@inline function priortogid(priority, mingid)
    (priority <= eps(typeof(priority))) && return 1
    gid = exponent(priority) + 1
    (gid <= mingid) && return 2
    return gid - mingid + 2
end

@inline function sample(pg::PriorityGroup, priorities, rng=Random.GLOBAL_RNG) 
    @unpack maxpriority, numpids, pids = pg    

    pididx = 0
    pid    = zero(eltype(pids))
    notdone = true

    @inbounds while true

        # pick a random element 
        r      = rand(rng) * numpids
        pididx = trunc(Int, r) 
        pid    = pids[pididx+1]

        # acceptance test
        ( (r - pididx)*maxpriority < priorities[pid] ) && break
    end

    pid
end

function sample(pt::PriorityTable, priorities, rng=Random.GLOBAL_RNG)
    @unpack groups, gsum, gsums = pt

    # return id zero if total priority is zero
    (gsum < eps(gsum)) && return zero(eltype(groups[1].pids))

    # sample a group, search from end (largest priorities)
    # NOTE, THIS ASSUMES THE FIRST PRIORITY IS ZERO!!!
    gid = length(gsums)
    @inbounds r = rand(rng) * gsum - gsums[gid]
    while r > zero(r)
        gid -= one(gid)
        @inbounds r -= gsums[gid] 
    end

    # sample element within the group
    @inbounds sample(groups[gid], priorities, rng)    
end

