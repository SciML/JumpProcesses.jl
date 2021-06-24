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
    generate_jumps!(p, u, params, t)
    nothing
end

#QUESTION do I need to pass in `u` if `integrator` is passed in?
# calculate the next jump / jump time
function generate_jumps!(p::NRMJumpAggregation, u, params, t)
    @unpack cur_rates = p

    p.next_jump_time, site = top_with_handle(p.pq)
    if rand(p.rng)*get_site_rate(cur_rates, site) < get_site_reactions_rate(cur_rates, site)
        #TODO linear search for the reaction
    else
        #TODO linear search for the species to diffuse
        #TODO linear search for the neighbor to diffuse to using neighbors function
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::NSMJumpAggregation, integrator, u, params, t)
    # execute jump
    u = update_state!(p, integrator, u)

    # update current jump rates and times
    update_dependent_rates_and_draw_new_firing_times!(p, u, params, t)
    nothing
end

"""
    update_state!(p, integrator)

updates state based on p.next_jump
"""
function update_state!(p, integrator)
    @unpack next_jump = p
    execute_spatial_jump!(p,integrator,next_jump)
    # save jump that was just exectued
    p.prev_jump = next_jump
    return integrator.u
end

######################## SSA specific helper routines ########################

function update_rates_after_jump!(p, u, t, jump::SpatialReaction)
    @unpack site, reaction_id = jump
    @inbounds dep_rxs = p.dep_gr[p.reaction_id]
    @unpack cur_rates, ma_jumps = p

    @inbounds for rx in dep_rxs
        rate = evalrxrate(u, reaction_id, ma_jumps)
        set_site_reaction_rate!(cur_rates, site, reaction_id, rate)
    end

    # draw new firing time for site
    site_rate = get_site_rate(cur_rates, site)
    if site_rate > zero(typeof(site_rate))
        update!(p.pq, site, t + randexp(p.rng) / rate)
    else
        update!(p.pq, site, typemax(t))
    end
end

function update_rates_after_jump!(p, u, t, jump::SpatialDiffusion)
    @unpack source_site, target_site, species_id = jump
    #TODO figure out which reactions depend on the species, update their rates in both sites, draw new times for both sites
end

# recalculate jump rates for jumps that depend on the just executed jump (p.next_jump)
function update_dependent_rates_and_draw_new_firing_times!(p::NSMJumpAggregation, u, t)
    @unpack jump = p
    update_rates_after_jump!(p, u, t, jump)
    end
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

######################## helper routines for all spatial SSAs ########################
function execute_spatial_jump!(p,integrator,jump::SpatialReaction)
    @unpack majumps = p
    @unpack site, reaction_id = jump
    #QUESTION what is SVector and does it matter for diffusion?
    if u_site isa SVector
        integrator.u[site] = executerx(integrator.u[site], reaction_id, ma_jumps)
    else
        executerx!(integrator.u[site], reaction_id, ma_jumps)
    end
end

function execute_spatial_jump!(p,integrator,jump::SpatialDiffusion)
    @unpack source_site, target_site, species_id = jump
    integrator.u[source_site][species_id] -= 1
    integrator.u[target_site][species_id] += 1
end
