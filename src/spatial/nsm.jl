# Implementation of the Next Subvolume Method on a grid


############################ NSM ###################################
struct NSM <: AbstractAggregatorAlgorithm end

mutable struct NSMJumpAggregation{J,T,R,C,S,RNG,DEPGR,PQ} # <: AbstractSpatialSSAJumpAggregator
    next_jump::J #some structure to identify the next event: reaction or diffusion
    prev_jump::J ##some structure to identify the previous event: reaction or diffusion
    next_jump_time::T
    end_time::T
    cur_rates::R #some structure to store current rates
    diffusion_constants::C #[[diffusion constants for site 1], [diffusion constants for site 2], ..., [diffusion constants for last site]]
    ma_jumps::S #massaction jumps
    # rates::F1 #rates for constant-rate jumps
    # affects!::F2 #affects! function determines the effect of constant-rate jumps
    save_positions::Tuple{Bool,Bool}
    rng::RNG
    dep_gr::DEPGR #dep graph is same for each locale
    pq::PQ
    spatial_system::AbstractSpatialSystem
end

function NSMJumpAggregation(nj::AbstractSpatialJump, njt::T, et::T, crs::R, diffusion_constants::C,
                                      maj::S, sps::Tuple{Bool,Bool},
                                      rng::RNG, spatial_system::AbstractSpatialSystem; num_specs, dep_graph=nothing, kwargs...) where {T,S,R,C,F1,F2,RNG}

    # a dependency graph is needed and must be provided if there are constant rate jumps
    if dep_graph === nothing
        dg = DiffEqJump.make_dependency_graph(num_specs, maj)
    else
        dg = dep_graph

        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    pq = MutableBinaryMinHeap{T}()

    #QUESTION is using `AbstractSpatialJump` here good style? How to do better?
    NSMJumpAggregation{AbstractSpatialJump,T,R,C,S,F1,F2,RNG,typeof(dg),typeof(pq)}(nj, nj, njt, et, crs, diffusion_constants, maj,
                                                            sps, rng, dg, pq, spatial_system)
end

+############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::NSM, num_species, end_time, diffusion_constants, ma_jumps, save_positions, rng, spatial_system; kwargs...)

    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(end_time)}(), Vector{Vector{Pair{Int,eltype(u[1])}}}(), Vector{Vector{Pair{Int,eltype(u[1])}}}())
    end

    next_jump = NoSpatialJump()
    next_jump_time = typemax(typeof(end_time))
    current_rates = SpatialRates(get_num_majumps(majumps), num_species, number_of_sites(spatial_system))

    NSMJumpAggregation(next_jump, next_jump_time, end_time, current_rates, diffusion_constants, majumps, save_positions, rng, spatial_system; num_specs = num_species, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
#QUESTION do I need to pass in integrator if I don't use it?
function initialize!(p::NSMJumpAggregation, integrator, u, params, t)
    fill_rates_and_get_times!(p, u, params, t)
    # TODO write specialized function to generate next jump (takes the top site and chooses the reaction/diffusion within it)
    generate_jumps!(p, u, params, t)
    nothing
end

#TODO finish this
#QUESTION do I need to pass in `u` if `integrator` is passed in?
# calculate the next jump / jump time
function generate_jumps!(p::NRMJumpAggregation, u, params, t)
    @unpack cur_rates = p

    p.next_jump_time, site = top_with_handle(p.pq)
    if rand(p.rng)*get_site_rate(cur_rates, site) < get_site_reactions_rate(cur_rates, site)
        #linear search for the reaction
    else
        #linear search for the species to diffuse
        #linear search for the neighbor to diffuse to using neighbors function
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::NSMJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)

    # update current jump rates and times
    # TODO write specialized functon to update the dependent rates (maybe a function to update in case of a reaction and another function to update in case of a diffusion)
    update_dependent_rates!(p, u, params, t)
    nothing
end

"""
    update_state!(p, integrator)

updates state based on p.next_jump
"""
function update_state!(p, integrator)
    #TODO
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


"""
reevaluate all rates, recalculate tentative site firing times, and reinit the priority queue
"""
function fill_rates_and_get_times!(aggregation::NRMJumpAggregation, u, t)
    @unpack majumps, cur_rates, diffusion_constants, spatial_system = aggregation
    @unpack reaction_rates, diffusion_rates = cur_rates
    num_sites = number_of_sites(spatial_system)
    num_majumps = get_num_majumps(majumps)
    num_species = length(u[1])

    @assert cur_rates.reaction_rates_sum == zeros(typeof(cur_rates.reaction_rates_sum[1]),num_sites)
    @assert cur_rates.diffusion_rates_sum == zeros(typeof(cur_rates.diffusion_rates_sum[1]),num_sites)

    pqdata = Vector{typeof(t)}(undef, num_sites)
    for site in 1:num_sites
        #reactions
        for rx in 1:num_majumps
            rate = evalrxrate(u[site], rx, majumps)
            set_site_reaction_rate!(cur_rates, site, rx, rate)
        end
        #diffusions
        for species in 1:num_species
            rate = diffusion_constants[site][species]
            set_site_diffusion_rate!(cur_rates, site, species, rate)
        end
        pqdata[site] = t + randexp(aggregation.rng) / get_site_reactions_rate(spatial_rates, site)
    end

    aggregation.pq = MutableBinaryMinHeap(pqdata)
    nothing
end
