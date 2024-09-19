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
mutable struct PriorityGroup{T, W <: AbstractVector}
    "(strict) upper bound for priorities in this group"
    maxpriority::T

    "number of priority ids"
    numpids::Int

    "priority ids associated with group"
    pids::W
end
PriorityGroup{U}(maxpriority::T) where {T, U} = PriorityGroup(maxpriority, 0, Vector{U}())

@inline function insert!(pg::PriorityGroup, pid)
    @unpack numpids, pids = pg

    if numpids == length(pids)
        push!(pids, pid)
    else
        @inbounds pids[numpids + 1] = pid
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

@inline function ids(pg::PriorityGroup)
    pg.pids[1:(pg.numpids)]
end

function Base.show(io::IO, pg::PriorityGroup)
    println(io, "  ", summary(pg))
    println(io, "  maxpriority = ", pg.maxpriority)
    println(io, "  numpids = ", pg.numpids)
    println(io, "  pids = ", ids(pg))
end

"""
Table to store the groups.
"""
mutable struct PriorityTable{F, S, T, U <: Function}
    "non-zero values below this are binned together, static"
    minpriority::F

    "values above this cause new groups to be added"
    maxpriority::F

    "bins storing priority ids within a given range"
    groups::Vector{PriorityGroup{F, Vector{S}}}

    "stores the sums of the priorities within each group"
    gsums::Vector{F}

    "stores the sum of the group sums"
    gsum::F

    "maps priority id to group and idx within the group"
    pidtogroup::Vector{Tuple{T, T}}

    "mapping from priority value to group id that stores it"
    priortogid::U
end

"""
Setup table from a vector of priorities. The id
of a priority is its position within this vector.
"""
function PriorityTable(priortogid::Function, priorities::AbstractVector, minpriority,
        maxpriority)
    numgroups = priortogid(maxpriority)
    numgroups -= one(typeof(numgroups))
    pidtype = typeof(numgroups)
    ptype = eltype(priorities)
    groups = Vector{PriorityGroup{ptype, Vector{pidtype}}}()
    pidtogroup = Vector{Tuple{Int, Int}}(undef, length(priorities))
    gsum = zero(ptype)
    gsums = zeros(ptype, numgroups)

    # create the groups, {0}, (0,minpriority), [minpriority,2*minpriority)...
    push!(groups, PriorityGroup{pidtype}(zero(ptype)))
    gmaxprior = minpriority
    for i in 2:numgroups
        push!(groups, PriorityGroup{pidtype}(gmaxprior))
        gmaxprior *= 2
    end
    #@assert 2*maxpriority <= gmaxprior "gmaxprior = $gmaxprior, maxpriority=$maxpriority"
    gmaxprior = groups[end].maxpriority

    pt = PriorityTable(minpriority, gmaxprior, groups, gsums, gsum, pidtogroup, priortogid)

    # insert priority ids into the groups
    for (pid, priority) in enumerate(priorities)
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
Adds extra groups to the table to accommodate a new maxpriority.
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
        @inbounds pidtogroup[pid] = (gid, pididx)
    else
        #@assert pid == length(pidtogroup) + 1
        push!(pidtogroup, (gid, pididx))
    end

    gid
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
    pdiff = newpriority - oldpriority
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
            pidtogroup[movedpid] = (oldgid, pididx)

            # insert the updated pid back and store it's new position in the group
            pididx = insert!(groups[newgid], pid)
            pidtogroup[pid] = (newgid, pididx)

            # update sums, special case if group empty to avoid FP error in running sums
            grpsz = groups[oldgid].numpids
            gsums[oldgid] = (grpsz == zero(grpsz)) ? zero(oldpriority) :
                            gsums[oldgid] - oldpriority
            gsums[newgid] += newpriority
        end
    end
    nothing
end


function reset!(pt::PriorityTable{F, S, T, U}) where {F, S, T, U}
    @unpack groups, gsums, pidtogroup = pt
    pt.gsum = zero(F)
    fill!(gsums, zero(F))
    fill!(pidtogroup, (zero(T), zero(T)))
    for group in groups
        group.numpids = zero(T)
    end
end

function Base.show(io::IO, pt::PriorityTable)
    println(io, summary(pt))
    println(io, "(minpriority,maxpriority) = ", (pt.minpriority, pt.maxpriority))
    println(io, "sum of priorities = ", pt.gsum)
    println(io, "num of groups = ", length(pt.groups))
    println(io, "pidtogroup = ", pt.pidtogroup)
    for (i, group) in enumerate(pt.groups)
        if group.numpids > 0
            println(io, "group = ", i, ", group sum = ", pt.gsums[i])
            Base.show(io, group)
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

@inline function sample(pg::PriorityGroup, priorities, rng = DEFAULT_RNG)
    @unpack maxpriority, numpids, pids = pg

    pididx = 0
    pid = zero(eltype(pids))

    @inbounds while true

        # pick a random element
        r = rand(rng) * numpids
        pididx = trunc(Int, r)
        pid = pids[pididx + 1]

        # acceptance test
        ((r - pididx) * maxpriority < priorities[pid]) && break
    end

    pid
end

function sample(pt::PriorityTable, priorities, rng = DEFAULT_RNG)
    @unpack groups, gsum, gsums = pt

    # return id zero if total priority is zero
    (gsum < eps(typeof(gsum))) && return zero(eltype(groups[1].pids))

    # sample a group, search from end (largest priorities)
    # NOTE, THIS ASSUMES THE FIRST PRIORITY IS ZERO!!!
    # gid = length(gsums)
    # @inbounds r = rand(rng) * gsum - gsums[gid]
    # while r > zero(r)
    #     gid -= one(gid)
    #     iszero(gid) && return gid   # if no result found return zero
    #     @inbounds r -= gsums[gid]
    # end
    gid = 0
    r = rand(rng) * gsum
    @inbounds for i in length(gsums):-1:1
        r -= gsums[i]
        if r <= zero(r)
            gid = i
            break
        end
    end
    iszero(gid) && return gid

    # sample element within the group
    @inbounds sample(groups[gid], priorities, rng)
end

##########################
### Routines for CCNRM ###
##########################

mutable struct TimeGrouper{T <: Number}
    mintime::T
    timestep::T
end

function (t::TimeGrouper)(time) 
    return floor(Int, (time - t.mintime) / t.timestep) + 1
end

mutable struct PriorityTimeTable{T <: Number, F <: Int}
    pt::PriorityTable
    # maxtime::T 
    # pidtogroup
    # 
    times::Vector{T}
    timegrouper::TimeGrouper
    minbin::F
    steps::F
end

# Construct the time table with the default optimal bin width and number of bins. 
# DEFAULT NUMBINS: 20 * √length(times)
# DEFAULT BINWIDTH: 16 / sum(propensities)
function PriorityTimeTable(times::AbstractVector, mintime, timestep) 
    numbins = floor(Int, 20 * sqrt(length(times)))
    maxtime = mintime + numbins * timestep

    pidtype = typeof(numbins)
    ptype = eltype(times)
    groups = Vector{PriorityGroup{ptype, Vector{pidtype}}}()
    pidtogroup = Vector{Tuple{Int, Int}}(undef, length(times))
    gsum = zero(ptype)
    gsums = zeros(ptype, numbins)

    ttgdata = TimeGrouper{ptype}(mintime, timestep)
    timetogroup = (time -> ttgdata(time))

    # Create the groups, [t_min, t_min + τ), [t_min + τ, t_min + 2τ)...
    for i in 1:numbins
        push!(groups, PriorityGroup{pidtype}(mintime + i*timestep))
    end

    pt = PriorityTable(mintime, maxtime, groups, gsums, gsum, pidtogroup, timetogroup)

    # Insert priority ids into the groups
    for (pid, time) in enumerate(times)
        if time > maxtime
            pidtogroup[pid] = (0, 0)
            continue
        end
        gid = insert!(pt, pid, time)
    end

    minbin = findfirst(g -> g.numpids >(0), pt.groups); minbin === nothing && (minbin = 0)
    ptt = PriorityTimeTable(pt, times, ttgdata, minbin, 0)
end

# Rebuild the table when there are no more reaction times within the current
# time window. 
function rebuild!(ptt::PriorityTimeTable, mintime, timestep)
    @unpack pt, times, timegrouper = ptt
    reset!(pt); groups = pt.groups

    numbins = floor(Int, 20 * sqrt(length(times)))
    pt.minpriority = mintime; pt.maxpriority = mintime + numbins*timestep
    timegrouper.mintime = mintime; timegrouper.timestep = timestep
    for i in 1:numbins
        groups[i].maxpriority = mintime + i*timestep
    end

    # Reinsert the times into the groups. 
    for (id, time) in enumerate(times)
        time > pt.maxpriority && continue
        insert!(pt, id, time)
    end
    ptt.minbin = findfirst(g -> g.numpids >(0), pt.groups); ptt.minbin === nothing && (ptt.minbin = 0)
    ptt.steps = 0
end

# Get the reaction with the earliest timestep.
function getfirst(ptt::PriorityTimeTable)
    @unpack pt, times, minbin = ptt
    minbin == 0 && return (nothing, nothing)
    groups = pt.groups; gsums = pt.gsums

    while groups[minbin].numpids == 0
        minbin += 1
        if minbin > length(groups)
            return (nothing, nothing) 
        end
    end
    
    ptt.minbin = minbin; ptt.steps += 1
    pids = ids(groups[minbin])
    min_time, min_idx = findmin(times[pids])
    return pids[min_idx], min_time
end

# Update the priority table when a reaction time gets updated. We only shift
# between bins if the new time is within the current time window; otherwise
# we remove the reaction and wait until rebuild. 
function update!(ptt::PriorityTimeTable, pid, oldtime, newtime) 
    @unpack pt, times, timegrouper = ptt
    maxtime = pt.maxpriority; pidtogroup = pt.pidtogroup; groups = pt.groups

    times[pid] = newtime
    if oldtime >= maxtime 
        # If a reaction comes back into the time window, insert it. 
        newtime < maxtime ? insert!(pt, pid, newtime) : return
    elseif newtime >= maxtime
        # If the new time lands outside of current window, remove it.
        pop!(pt, pid, oldtime) 
    else
        # Move bins if the reaction was already inside. 
        move_bins!(pt, pid, oldtime, newtime) 
    end
end

function pop!(pt::PriorityTable, pid, oldtime) 
    @unpack pidtogroup, groups = pt
    @inbounds begin 
        gid, pidx = pidtogroup[pid]
        movedpid = remove!(groups[gid], pidx)
        pidtogroup[movedpid] = (gid, pidx)
        pidtogroup[pid] = (0, 0)
    end
    gid
end

function move_bins!(pt::PriorityTable, pid, oldtime, newtime) 
    @unpack pidtogroup, groups, priortogid = pt 
    oldgid = priortogid(oldtime); newgid = priortogid(newtime)
    oldgid == newgid && return 
    @inbounds begin
        rank = pidtogroup[pid][2]
        movedpid = remove!(groups[oldgid], rank)
        pidtogroup[movedpid] = (oldgid, rank)
        newrank = insert!(groups[newgid], pid)
        pidtogroup[pid] = (newgid, newrank)
    end
end
