# Implementation of the Next Subvolume Method on a grid


############################ NSM ###################################
struct NSM <: AbstractAggregatorAlgorithm end

mutable struct NSMJumpAggregation{J,T,R,S,RNG,DEPGR,PQ} # <: AbstractSpatialSSAJumpAggregator
    next_jump::J #some structure to identify the next event: reaction or diffusion
    prev_jump::J ##some structure to identify the previous event: reaction or diffusion
    next_jump_time::T
    end_time::T
    cur_rates::R #some structure to store current rates
    ma_jumps::S #massaction jumps
    # rates::F1 #rates for constant-rate jumps
    # affects!::F2 #affects! function determines the effect of constant-rate jumps
    save_positions::Tuple{Bool,Bool}
    rng::RNG
    dep_gr::DEPGR #dep graph is same for each locale
    pq::PQ
    spatial_system::AbstractSpatialSystem
end

function NSMJumpAggregation(nj::AbstractSpatialJump, njt::T, et::T, crs::R,
                                      maj::S, sps::Tuple{Bool,Bool},
                                      rng::RNG, spatial_system::AbstractSpatialSystem; num_specs, dep_graph=nothing, kwargs...) where {T,S,R,F1,F2,RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph === nothing
        dg = make_dependency_graph(num_specs, maj)
    else
        dg = dep_graph

        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    pq = MutableBinaryMinHeap{T}()

    # TODO is using `AbstractSpatialJump` here good style? How to do better?
    NSMJumpAggregation{AbstractSpatialJump,T,R,S,F1,F2,RNG,typeof(dg),typeof(pq)}(nj, nj, njt, et, crs, maj,
                                                            sps, rng, dg, pq, spatial_system)
end

+############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::NSM, u, end_time, ma_jumps, save_positions, rng, spatial_system; kwargs...)

    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(end_time)}(), Vector{Vector{Pair{Int,eltype(u[1])}}}(), Vector{Vector{Pair{Int,eltype(u[1])}}}())
    end

    num_species = length(u[1])
    next_jump = NoSpatialJump()
    next_jump_time = typemax(typeof(end_time))
    current_rates = SpatialRates(get_num_majumps(majumps), num_species, number_of_sites(spatial_system))

    # TODO how should this works?
    NSM(next_jump, next_jump_time, end_time, current_rates, majumps, save_positions, rng, spatial_system; num_specs = num_species, kwargs...)
end



# set up a new simulation and calculate the first jump / jump time
function initialize!(p::NSMJumpAggregation, integrator, u, params, t)
    # TODO write specialized function to fill in the current rates data structure
    # TODO write specialized function to get tentative firing times for all sites from the current rates data strcture
    fill_rates_and_get_times!(p, u, params, t)
    # TODO write specialized function to generate next jump (takes the top site and chooses the reaction/diffusion within it)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::NSMJumpAggregation, integrator, u, params, t)
    # execute jump
    # TODO write specialized function to update the current state (need to decide how the state is represented)
    u = update_state!(p, integrator, u)

    # update current jump rates and times
    # TODO write specialized functon to update the dependent rates (maybe a function to update in case of a reaction and another function to update in case of a diffusion)
    update_dependent_rates!(p, u, params, t)
    nothing
end

# calculate the next jump / jump time
# just the top of the priority queue
function generate_jumps!(p::NSMJumpAggregation, integrator, u, params, t)
    # TODO write specialized function to choose the next jump
    p.next_jump_time, p.next_jump = top_with_handle(p.pq)
    nothing
end


######################## SSA specific helper routines ########################

# recalculate jump rates for jumps that depend on the just executed jump (p.next_jump)
function update_dependent_rates!(p::NSMJumpAggregation, u, params, t)
    @inbounds dep_rxs = p.dep_gr[p.next_jump]
    @unpack cur_rates, rates, ma_jumps = p
    num_majumps = get_num_majumps(ma_jumps)

    @inbounds for rx in dep_rxs
        oldrate = cur_rates[rx]

        # update the jump rate
        @inbounds cur_rates[rx] = calculate_jump_rate(ma_jumps, num_majumps, rates, u, params, t, rx)

        # calculate new jump times for dependent jumps
        if rx != p.next_jump && oldrate > zero(oldrate)
            if cur_rates[rx] > zero(eltype(cur_rates))
                update!(p.pq, rx, t + oldrate / cur_rates[rx] * (p.pq[rx] - t))
            else
                update!(p.pq, rx, typemax(t))
            end
        else
            if cur_rates[rx] > zero(eltype(cur_rates))
                update!(p.pq, rx, t + randexp(p.rng) / cur_rates[rx])
            else
                update!(p.pq, rx, typemax(t))
            end
        end

    end
    nothing
end


# reevaulate all rates, recalculate all jump times, and reinit the priority queue
function fill_rates_and_get_times!(p::NSMJumpAggregation, u, params, t)

    # mass action jumps
    majumps   = p.ma_jumps
    cur_rates = p.cur_rates
    pqdata = Vector{typeof(t)}(undef,length(cur_rates))
    @inbounds for i in 1:get_num_majumps(majumps)
        cur_rates[i] = evalrxrate(u, i, majumps)
        pqdata[i] = t + randexp(p.rng) / cur_rates[i]
    end

    # constant rates
    rates = p.rates
    idx   = get_num_majumps(majumps) + 1
    @inbounds for rate in rates
        cur_rates[idx] = rate(u, params, t)
        pqdata[idx] = t + randexp(p.rng) / cur_rates[idx]
        idx += 1
    end

    # setup a new indexed priority queue to storing rx times
    p.pq = MutableBinaryMinHeap(pqdata)
    nothing
end


function build_spatial_jump_aggregation(NSMJumpAggregation, u, p, t, end_time, ma_jumps,
                       save_positions, rng; num_specs=length(u), kwargs...)
    # mass action jumps
    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(t)}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}())
    end
    cur_rates = SpatialRates{typeof()}
end

function build_jump_aggregation(jump_agg_type, u, p, t, end_time, ma_jumps, rates,
                                affects!, save_positions, rng; kwargs...)

    # mass action jumps
    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(t)}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}(),
                             Vector{Vector{Pair{Int,eltype(u)}}}())
    end

    # current jump rates, allows mass action rates
    # TODO: make a structure for current rates and initialize it here
    cur_rates = Vector{typeof(t)}(undef, get_num_majumps(majumps) + length(rates))

    sum_rate = zero(typeof(t))
    next_jump = 0
    next_jump_time = typemax(typeof(t))
    jump_agg_type(next_jump, next_jump_time, end_time, cur_rates, sum_rate,
                majumps, rates, affects!, save_positions, rng; kwargs...)
end
