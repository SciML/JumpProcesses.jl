# Implementation of the Next Subvolume Method on a grid


############################ NSM ###################################
# NOTE make 0 a sink state for absorbing boundary condition

#NOTE state vector u is a matrix. u[i,j] is species i, site j
#NOTE hopping_constants is a matrix. hopping_constants[i,j] is species i, site j
mutable struct NSMJumpAggregation{J,T,R<:AbstractSpatialRates,C,S,RNG,DEPGR,VJMAP,JVMAP,PQ,SS} <: AbstractSSAJumpAggregator
    next_jump::SpatialJump{J} #some structure to identify the next event: reaction or hop
    prev_jump::SpatialJump{J} #some structure to identify the previous event: reaction or hop
    next_jump_time::T
    end_time::T
    cur_rates::R #some structure to store current rates
    hopping_constants::C #matrix with ith column being hop constants for site i
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

function NSMJumpAggregation(nj::SpatialJump{J}, njt::T, et::T, crs::R, hopping_constants::C,
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

    NSMJumpAggregation{J,T,R,C,S,RNG,typeof(dg),typeof(vtoj_map),typeof(jtov_map),typeof(pq),SS}(nj, nj, njt, et, crs, hopping_constants, maj, sps, rng, dg, vtoj_map, jtov_map, pq, spatial_system)
end

############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::NSM, starting_state, p, t, end_time, constant_jumps, ma_jumps, save_positions, rng; hopping_constants, spatial_system, kwargs...)
    num_species = size(starting_state,1)
    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(end_time)}(), Vector{Vector{Pair{Int,Int}}}(), Vector{Vector{Pair{Int,Int}}}())
    end

    next_jump = SpatialJump{Int}(typemax(Int),typemax(Int),typemax(Int)) #a placeholder
    next_jump_time = typemax(typeof(end_time))
    current_rates = SpatialRates(get_num_majumps(majumps), num_species, num_sites(spatial_system))

    NSMJumpAggregation(next_jump, next_jump_time, end_time, current_rates, hopping_constants, majumps, save_positions, rng, spatial_system; num_specs = num_species, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::NSMJumpAggregation, integrator, u, params, t)
    fill_rates_and_get_times!(p, u, t)
    generate_jumps!(p, integrator, params, u, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::NSMJumpAggregation, integrator, params, u, t)
    @unpack cur_rates, rng = p

    p.next_jump_time, site = top_with_handle(p.pq)
    if rand(rng)*total_site_rate(cur_rates, site) < total_site_rx_rate(cur_rates, site)
        rx = linear_search(rx_rates_at_site(cur_rates, site), rand(rng) * total_site_rx_rate(cur_rates, site))
        p.next_jump = SpatialJump(site, rx+size(p.hopping_constants, 1), site)
    else
        species_to_diffuse = linear_search(hop_rates_at_site(cur_rates, site), rand(rng) * total_site_hop_rate(cur_rates, site))
        nbs = neighbors(p.spatial_system, site)
        target_site = nbs[rand(rng,1:num_neighbors(p.spatial_system, site))] # random neighbor
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
    @unpack ma_jumps, hopping_constants, spatial_system, cur_rates = aggregation

    reset!(cur_rates)
    num_majumps = get_num_majumps(ma_jumps)
    num_species = size(u,1) #NOTE assumes u is a matrix with ith column being the ith site
    num_sites = DiffEqJump.num_sites(spatial_system)

    pqdata = Vector{typeof(t)}(undef, num_sites)
    for site in 1:num_sites
        update_reaction_rates!(cur_rates, 1:num_majumps, u, ma_jumps, site)
        update_hop_rates!(cur_rates, 1:num_species, hopping_constants, u, site, spatial_system)
        pqdata[site] = t + randexp(aggregation.rng) / total_site_rate(cur_rates, site)
    end

    aggregation.pq = MutableBinaryMinHeap(pqdata)
    nothing
end

"""
    update_dependent_rates_and_firing_times!(p, u, t)

recalculate jump rates for jumps that depend on the just executed jump (p.prev_jump)
"""
function update_dependent_rates_and_firing_times!(p, u, t)
    jump = p.prev_jump
    if is_hop(p, jump)
        source_site = jump.src
        target_site = jump.dst
        update_rates_after_hop!(p, u, source_site, target_site, jump.jidx)
        update_site_time!(p.pq, p.rng, p.cur_rates, source_site, t)
        update_site_time!(p.pq, p.rng, p.cur_rates, target_site, t)
    else
        site = jump.src
        update_rates_after_reaction!(p, u, site, reaction_id_from_jump(p,jump))
        update_site_time!(p.pq, p.rng, p.cur_rates, site, t)
    end
end

function update_site_time!(pq, rng, cur_rates, site, t)
    site_rate = total_site_rate(cur_rates, site)
    if site_rate > zero(typeof(site_rate))
        update!(pq, site, t + randexp(rng) / site_rate)
    else
        update!(pq, site, typemax(t))
    end
end

######################## helper routines for all spatial SSAs ########################
function update_rates_after_reaction!(p, u, site, reaction_id)
    update_reaction_rates!(p.cur_rates, p.dep_gr[reaction_id], u, p.ma_jumps, site)
    update_hop_rates!(p.cur_rates, p.jumptovars_map[reaction_id], p.hopping_constants, u, site, p.spatial_system)
end

function update_rates_after_hop!(p, u, source_site, target_site, species)
    update_reaction_rates!(p.cur_rates, p.vartojumps_map[species], u, p.ma_jumps, source_site)
    update_hop_rates!(p.cur_rates, species, p.hopping_constants, u, source_site, p.spatial_system)
    
    update_reaction_rates!(p.cur_rates, p.vartojumps_map[species], u, p.ma_jumps, target_site)
    update_hop_rates!(p.cur_rates, species, p.hopping_constants, u, target_site, p.spatial_system)
end

"""
update rates of all reactions in rxs at site
"""
function update_reaction_rates!(cur_rates, rxs, u, ma_jumps, site)
    for rx in rxs
        set_rx_rate_at_site!(cur_rates, site, rx, evalrxrate((@view u[:,site]), rx, ma_jumps))
    end
end

"""
update rates of all specs in list species at site
"""
function update_hop_rates!(cur_rates, species::AbstractArray, hopping_constants, u, site, spatial_system)
    for spec in species
        set_hop_rate_at_site!(cur_rates, site, spec, evalhoppingrate(hopping_constants, u, spec, site, spatial_system))
    end
end
"""
update rates of species at site
"""
function update_hop_rates!(cur_rates, species, hopping_constants, u, site, spatial_system)
    set_hop_rate_at_site!(cur_rates, site, species, evalhoppingrate(hopping_constants, u, species, site, spatial_system))
end

"""
update_state!(p, integrator)

updates state based on p.next_jump
"""
function update_state!(p, integrator)
    jump = p.next_jump
    if is_hop(p, jump)
        execute_hop!(integrator, jump.src, jump.dst, jump.jidx)
    else
        rx_index = reaction_id_from_jump(p,jump)
        executerx!((@view integrator.u[:,jump.src]), rx_index, p.ma_jumps)
    end
    # save jump that was just exectued
    p.prev_jump = jump
    nothing
end

"""
    is_hop(p, jump)

true if jump is a hop
"""
function is_hop(p, jump)
    jump.jidx <= size(p.hopping_constants,1)
end

"""
    execute_hop!(integrator, jump)

documentation
"""
function execute_hop!(integrator, source_site, target_site, species)
    integrator.u[species,source_site] -= 1
    integrator.u[species,target_site] += 1
end

"""
    reaction_id_from_jump(p,jump)

return reaction id by subtracting the number of hops
"""
function reaction_id_from_jump(p,jump)
    jump.jidx - size(p.hopping_constants,1)
end

"""
    evalhoppingrate(args)

documentation
"""
function evalhoppingrate(hopping_constants, u, species, site, spatial_system)
    u[species,site]*hopping_constants[species,site]*num_neighbors(spatial_system, site)
end

"""
number of constant rate jumps
"""
num_constant_rate_jumps(aggregator::NSMJumpAggregation) = 0
