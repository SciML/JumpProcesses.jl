"""
 Dynamic table data structure to store and update priorities

 Implementation
 Stores ids that identify the priorities. Each group stores a range of priority ids. 
 The basic design assumes a lower-most group of "zero" priorities, and a second
 group storing all non-zero priorities that are <= a `minpriority`. All other groups
 store priorities within consecutive ranges. The `minpriority` is taken fixed at creation,
 but the `maxpriority` will increase dynamically based on inserted / updated priorities.
 The ranges are assumed to be powers of two:
    bin 1 = {0}, 
    bin 2 = (0,`minpriority`], 
    bin 3 = (`minpriority`,`2*minpriority`]...
    bin N = (``.5*maxpriority`,`maxpriority`]
"""

"""
One group (i.e. bin) of priority ids within the table
"""
mutable struct PriorityGroup{T,W <: AbstractVector}
    "upper bound for priorities in this group"
    maxpriority::T

    "number of priorities"
    numids::Int

    "ids associated with the priorities"
    ids::W
end
PriorityGroup{U}(maxpriority::T) where {T,U} = 
                PriorityGroup(maxpriority, 0, Vector{U}())

@inline function insert!(pg::PriorityGroup, id)
    @unpack numids, ids = pg

    if numids == length(ids)
        push!(ids, id)
    else
        @inbounds ids[numids+1] = id
    end

    # return the position of insertion
    pg.numids += 1
end

@inline function remove!(pg::PriorityGroup, ididx)
    @unpack numids, ids = pg

    # simply swap the last id with the one to remove
    @inbounds ids[ididx] = ids[numids]
    pg.numids -= 1
    nothing
end


"""
Table to store the groups.
"""
mutable struct PriorityTable{F,S,T,U<:Function}
    "non-zero values below this are binned together, static"
    minpriority::F   

    "values above this cause new groups to be added"
    maxpriority::F   

    groups::Vector{PriorityGroup{F,Vector{S}}}

    "stores the sums of the priorities within each group"
    gsums::Vector{F}

    "stores the sum of the group sums"
    gsum::F

    "maps priority id to group and idx within the group"
    idtogroup::Vector{Tuple{T,T}}

    "mapping from priority value to group id that stores it"
    priortoid::U
end

# Setup table from a vector of priorities. The id
# of a priority is its position within this vector.
function PriorityTable(priortoid::Function, priorities::AbstractVector, minpriority, maxpriority)

    numgroups = priortoid(maxpriority)
    pidtype   = typeof(numgroups)
    ptype     = eltype(priorities)
    idtogroup = Vector{pidtype}()
    gsum      = zero(ptype)
    gsums     = zeros(ptype, numgroups)

    # create the groups, {0}, (0,minpriority], (minpriority,2*minpriority]...
    push!(idtogroup, PriorityGroup{pidtype}(zero(ptype)))
    gmaxprior = minpriority
    for i = 2:numgroups
        push!(idtogroup, PriorityGroup{pidtype}(gmaxprior))
        gmaxprior = 2*gmaxprior
    end

    # insert priority ids into the groups
    for (pid,priority) in enumerate(priorities)
        gid         = priortoid(priority)
        gsum       += priority
        gsums[gid] += priority
        insert!(idtogroup[gid], pid)
    end



end

function rebuild!(pt::PriorityTable, newminPriority, newmaxPriority)
    nothing
end

function insert!(pt::PriorityTable, jumpid, Priority)
    
    # add a new bucket if Priority is too big
    if Priority > pt.maxPriority

    else
        id = pt.idfun(Priority)
    end 

    nothing
end

function update!(pt::PriorityTable, jumpid, oldPriority, newPriority)

    nothing
end


#######################################################
# sampling routines for DirectCR
#######################################################

@inline function sample(p::DirectCRJumpAggregation, pg::PriorityGroup, rng=Random.GLOBAL_RNG) 
    @unpack maxpriority, numids, ids = pg
    priorities = p.cur_rates

    ididx = 0
    id    = zero(eltype(ids))
    notdone = true
    @inbounds while notdone

        # pick a random element 
        r     = rand(rng) * numids
        ididx = trunc(Int, r) 
        id    = ids[ididx]

        # acceptance test
        if (r - ididx)*maxpriority < priorities[id]
            notedone = false
        end
    end

    id
end

function sample(p::DirectCRJumpAggregation, pt::PriorityTable, rng=Random.GLOBAL_RNG)
    @unpack groups, gsums = pt
    maxpriority = p.sum_rate

    # sample a group, search from end (lapgest priorities)
    # NOTE, THIS ASSUMES THE FIRST PRIORITY IS ZERO!!!
    r     = rand(rng) * maxpriority
    gid   = length(gsum)
    rtsum = maxpriority - gsums[gid]
    while rtsum > r
        gid   -= one(gid)
        rtsum -= gsums[gid] 
    end

    # sample element within the group
    sample(p, groups[gid], rng)    
end

