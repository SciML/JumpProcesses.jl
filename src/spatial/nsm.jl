# Implementation of the Next Subvolume Method on a grid


############################ NSM ###################################
# NOTE make 0 a sink state for absorbing boundary condition

#NOTE state vector u is a matrix. u[i,j] is species i, site j
#NOTE diffusion_constants is a matrix. diffusion_constants[i,j] is species i, site j
mutable struct NSMJumpAggregation{J,T,R<:AbstractSpatialRates,C,S,RNG,DEPGR,VJMAP,JVMAP,PQ,SS<:AbstractSpatialSystem} <: AbstractSSAJumpAggregator
    next_jump::SpatialJump{J} #some structure to identify the next event: reaction or diffusion
    prev_jump::SpatialJump{J} #some structure to identify the previous event: reaction or diffusion
    next_jump_time::T
    end_time::T
    cur_rates::R #some structure to store current rates
    diffusion_constants::C #matrix with ith column being diffusion constants for site i
    ma_jumps::S #massaction jumps
    # rates::F1 #rates for constant-rate jumps
    # affects!::F2 #affects! function determines the effect of constant-rate jumps
    save_positions::Tuple{Bool,Bool}
    rng::RNG
    dep_gr::DEPGR #dep graph is same for each site
    vartojumps_map::VJMAP #vartojumps_map is same for each site
    jumptovars_map::JVMAP #jumptovars_map is same for each site
    pq::PQ
    spatial_system::SS
end

#TODO go through and see if I need to slice or use @view

function NSMJumpAggregation(nj::SpatialJump{J}, njt::T, et::T, crs::R, diffusion_constants::C,
                                      maj::S, sps::Tuple{Bool,Bool},
                                      rng::RNG, spatial_system::SS; num_specs, vartojumps_map=nothing, jumptovars_map=nothing, dep_graph=nothing, kwargs...) where {J,T,S,R,C,RNG,SS}

    # a dependency graph is needed
    if dep_graph === nothing
        dg = DiffEqJump.make_dependency_graph(num_specs, maj)
    else
        dg = dep_graph
        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    # a species-to-reactions graph is needed
    if vartojumps_map === nothing
        vtoj_map = var_to_jumps_map(num_specs, maj)
    else
        vtoj_map = vartojumps_map
    end

    if jumptovars_map === nothing
        jtov_map = jump_to_vars_map(maj)
    else
        jtov_map = jumptovars_map
    end

    pq = MutableBinaryMinHeap{T}()

    NSMJumpAggregation{J,T,R,C,S,RNG,typeof(dg),typeof(vtoj_map),typeof(jtov_map),typeof(pq),SS}(nj, nj, njt, et, crs, diffusion_constants, maj, sps, rng, dg, vtoj_map, jtov_map, pq, spatial_system)
end

############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::NSM, num_species, end_time, diffusion_constants, ma_jumps, save_positions, rng, spatial_system; kwargs...)

    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(end_time)}(), Vector{Vector{Pair{Int,Int}}}(), Vector{Vector{Pair{Int,Int}}}())
    end

    next_jump = SpatialJump{Int}(typemax(Int),typemax(Int),typemax(Int)) #a placeholder
    next_jump_time = typemax(typeof(end_time))
    current_rates = SpatialRates(get_num_majumps(majumps), num_species, number_of_sites(spatial_system))

    NSMJumpAggregation(next_jump, next_jump_time, end_time, current_rates, diffusion_constants, majumps, save_positions, rng, spatial_system; num_specs = num_species, kwargs...)
end

function aggregate(aggregator::NSM, starting_state, p, t, end_time, constant_jumps, ma_jumps, save_positions, rng; diffusion_constants, spatial_system, kwargs...)
    num_species = length(@view starting_state[:,1])
    aggregate(aggregator, num_species, end_time, diffusion_constants, ma_jumps, save_positions, rng, spatial_system; kwargs...)
end

#NOTE integrator and params are not used. They remain to adhere to the interface of `AbstractSSAJumpAggregator` defined in ssajump.jl
# set up a new simulation and calculate the first jump / jump time
function initialize!(p::NSMJumpAggregation, integrator, u, params, t)
    fill_rates_and_get_times!(p, u, t)
    generate_jumps!(p, integrator, params, u, t)
    nothing
end

#NOTE integrator and params are not used. They remain to adhere to the interface of `AbstractSSAJumpAggregator` defined in ssajump.jl
# calculate the next jump / jump time
function generate_jumps!(p::NSMJumpAggregation, integrator, params, u, t)
    @unpack cur_rates, rng = p

    p.next_jump_time, site = top_with_handle(p.pq)
    if rand(rng)*get_site_rate(cur_rates, site) < get_site_reactions_rate(cur_rates, site)
        rx = linear_search(get_site_reactions_iterator(cur_rates, site), rand(rng) * get_site_reactions_rate(cur_rates, site))
        p.next_jump = SpatialJump(site, rx+length(@view p.diffusion_constants[:,site]), site)
    else
        species_to_diffuse = linear_search(get_site_diffusions_iterator(cur_rates, site), rand(rng) * get_site_diffusions_rate(cur_rates, site))
        #TODO this is not efficient. We iterate over neighbors twice.
        n = rand(rng,1:num_neighbors(p.spatial_system, site))
        target_site = nth_neighbor(p.spatial_system,site,n)
        p.next_jump = SpatialJump(site, species_to_diffuse, target_site)
    end
end

# execute one jump, changing the system state
function execute_jumps!(p::NSMJumpAggregation, integrator, u, params, t)
    # execute jump
    update_state!(p, integrator)

    # update current jump rates and times
    update_dependent_rates_and_firing_times!(p, integrator.u, t)
    nothing
end

######################## SSA specific helper routines ########################
"""
reevaluate all rates, recalculate tentative site firing times, and reinit the priority queue
"""
function fill_rates_and_get_times!(aggregation::NSMJumpAggregation, u, t)
    @unpack ma_jumps, diffusion_constants, spatial_system = aggregation

    num_majumps = get_num_majumps(ma_jumps)
    num_species = length(@view u[:,1]) #NOTE assumes u is a matrix with ith column being the ith site
    num_sites = number_of_sites(spatial_system)
    cur_rates = SpatialRates(num_majumps,num_species,num_sites)

    @assert cur_rates.reaction_rates_sum == zeros(typeof(cur_rates.reaction_rates_sum[1]),num_sites)
    @assert cur_rates.diffusion_rates_sum == zeros(typeof(cur_rates.diffusion_rates_sum[1]),num_sites)

    pqdata = Vector{typeof(t)}(undef, num_sites)
    for site in 1:num_sites
        update_reaction_rates!(cur_rates, 1:num_majumps, u, ma_jumps, site)
        update_diffusion_rates!(cur_rates, 1:num_species, diffusion_constants, u, site, spatial_system)
        pqdata[site] = t + randexp(aggregation.rng) / get_site_rate(cur_rates, site)
    end

    aggregation.cur_rates = cur_rates
    aggregation.pq = MutableBinaryMinHeap(pqdata)
    nothing
end

"""
    update_dependent_rates_and_firing_times!(p, u, t)

recalculate jump rates for jumps that depend on the just executed jump (p.prev_jump)
"""
function update_dependent_rates_and_firing_times!(p, u, t)
    jump = p.prev_jump
    if is_diffusion(p, jump)
        source_site = jump.site
        target_site = jump.target_site
        update_rates_after_diffusion!(p, u, t, source_site, target_site, jump.index)
        for site in [source_site, target_site]
            update_site_time!(p.pq, p.rng, p.cur_rates, site, t)
        end
    else
        site = jump.site
        update_rates_after_reaction!(p, u, t, site, reaction_id_from_jump(p,jump))
        update_site_time!(p.pq, p.rng, p.cur_rates, site, t)
    end
end

function update_site_time!(pq, rng, cur_rates, site, t)
    site_rate = get_site_rate(cur_rates, site)
    if site_rate > zero(typeof(site_rate))
        update!(pq, site, t + randexp(rng) / site_rate)
    else
        update!(pq, site, typemax(t))
    end
end

######################## helper routines for all spatial SSAs ########################
function update_rates_after_reaction!(p, u, t, site, reaction_id)
    update_reaction_rates!(p.cur_rates, p.dep_gr[reaction_id], u, p.ma_jumps, site)
    update_diffusion_rates!(p.cur_rates, p.jumptovars_map[reaction_id], p.diffusion_constants, u, site, p.spatial_system)
end

function update_rates_after_diffusion!(p, u, t, source_site, target_site, species)
    for site in [source_site, target_site]
        update_reaction_rates!(p.cur_rates, p.vartojumps_map[species], u, p.ma_jumps, site)
        update_diffusion_rates!(p.cur_rates, [species], p.diffusion_constants, u, site, p.spatial_system)
    end
end

"""
update rates of all reactions in rxs at site
"""
function update_reaction_rates!(cur_rates, rxs, u, ma_jumps, site)
    for rx in rxs
        set_site_reaction_rate!(cur_rates, site, rx, evalrxrate((@view u[:,site]), rx, ma_jumps))
    end
end

"""
update rates of all specs in list species at site
"""
function update_diffusion_rates!(cur_rates, species, diffusion_constants, u, site, spatial_system)
    for spec in species
        set_site_diffusion_rate!(cur_rates, site, spec, evaldiffrate(diffusion_constants, u, spec, site, spatial_system))
    end
end

"""
update_state!(p, integrator)

updates state based on p.next_jump
"""
function update_state!(p, integrator)
    jump = p.next_jump
    if is_diffusion(p, jump)
        execute_diffusion!(integrator, jump.site, jump.target_site, jump.index)
    else
        # u_site = integrator.u[:,jump.site]
        rx_index = reaction_id_from_jump(p,jump)
        executerx!((@view integrator.u[:,jump.site]), rx_index, p.ma_jumps)
        # executerx!(u_site, rx_index, p.ma_jumps)
        #QUESTION why does this not happen in-place?
        # integrator.u[:,jump.site] = u_site
    end
    # save jump that was just exectued
    p.prev_jump = jump
    nothing
end

"""
    is_diffusion(p, jump)

true if jump is a diffusion
"""
function is_diffusion(p, jump)
    # size(p.diffusion_constants,1)
    jump.index <= length(@view p.diffusion_constants[:,jump.site])
end

"""
    execute_diffusion!(integrator, jump)

documentation
"""
function execute_diffusion!(integrator, source_site, target_site, species)
    integrator.u[species,source_site] -= 1
    integrator.u[species,target_site] += 1
end

"""
    reaction_id_from_jump(p,jump)

return reaction id by subtracting the number of diffusive hops
"""
function reaction_id_from_jump(p,jump)
    jump.index - length(@view p.diffusion_constants[:,jump.site])
end

"""
    evaldiffrate(args)

documentation
"""
function evaldiffrate(diffusion_constants, u, species, site, spatial_system)
    u[species,site]*diffusion_constants[species,site]*num_neighbors(spatial_system, site)
end

"""
number of constant rate jumps
"""
num_constant_rate_jumps(aggregator::NSMJumpAggregation) = 0
