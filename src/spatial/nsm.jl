# Implementation of the Next Subvolume Method


############################ NSM ###################################
#NOTE state vector u is a matrix. u[i,j] is species i, site j
#NOTE hopping_constants is a matrix. hopping_constants[i,j] is species i, site j
mutable struct NSMJumpAggregation{J,T,RX,HOP,RNG,DEPGR,VJMAP,JVMAP,PQ,SS} <: AbstractSSAJumpAggregator
    next_jump::SpatialJump{J} #some structure to identify the next event: reaction or hop
    prev_jump::SpatialJump{J} #some structure to identify the previous event: reaction or hop
    next_jump_time::T
    end_time::T
    rx_rates::RX
    hop_rates::HOP
    # rates::F1 #rates for constant-rate jumps
    # affects!::F2 #affects! function determines the effect of constant-rate jumps
    save_positions::Tuple{Bool,Bool}
    rng::RNG
    dep_gr::DEPGR #dep graph is same for each site
    vartojumps_map::VJMAP #vartojumps_map is same for each site
    jumptovars_map::JVMAP #jumptovars_map is same for each site
    pq::PQ
    spatial_system::SS
    numspecies::Int #number of species
end

function NSMJumpAggregation(nj::SpatialJump{J}, njt::T, et::T, rx_rates::RX, hop_rates::HOP, sps::Tuple{Bool,Bool},
                                      rng::RNG, spatial_system::SS; num_specs, vartojumps_map=nothing, jumptovars_map=nothing, dep_graph=nothing, kwargs...) where {J,T,RX,HOP,RNG,SS}

    # a dependency graph is needed
    if dep_graph === nothing
        dg = DiffEqJump.make_dependency_graph(num_specs, rx_rates.ma_jumps)
    else
        dg = dep_graph
        # make sure each jump depends on itself
        add_self_dependencies!(dg)
    end

    # a species-to-reactions graph is needed
    if vartojumps_map === nothing
        vtoj_map = var_to_jumps_map(num_specs, rx_rates.ma_jumps)
    else
        vtoj_map = vartojumps_map
    end

    if jumptovars_map === nothing
        jtov_map = jump_to_vars_map(rx_rates.ma_jumps)
    else
        jtov_map = jumptovars_map
    end

    pq = MutableBinaryMinHeap{T}()

    NSMJumpAggregation{J,T,RX,HOP,RNG,typeof(dg),typeof(vtoj_map),typeof(jtov_map),typeof(pq),SS}(nj, nj, njt, et, rx_rates, hop_rates, sps, rng, dg, vtoj_map, jtov_map, pq, spatial_system, num_specs)
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
    rx_rates = RxRates(num_sites(spatial_system), majumps)
    hop_rates = HopRates(hopping_constants)

    NSMJumpAggregation(next_jump, next_jump_time, end_time, rx_rates, hop_rates, save_positions, rng, spatial_system; num_specs = num_species, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::NSMJumpAggregation, integrator, u, params, t)
    fill_rates_and_get_times!(p, u, t)
    generate_jumps!(p, integrator, params, u, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::NSMJumpAggregation, integrator, params, u, t)
    @unpack rx_rates, hop_rates, rng = p

    p.next_jump_time, site = top_with_handle(p.pq)
    if rand(rng)*(total_site_rx_rate(rx_rates, site)+total_site_hop_rate(hop_rates, site)) < total_site_rx_rate(rx_rates, site)
        rx = sample_rx_at_site(rx_rates, site, rng)
        p.next_jump = SpatialJump(site, rx+p.numspecies, site)
    else
        species_to_diffuse, target_site = sample_hop_at_site(hop_rates, site, rng, p.spatial_system)
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
    @unpack spatial_system, rx_rates, hop_rates = aggregation

    reset!(rx_rates)
    reset!(hop_rates)

    num_majumps = get_num_majumps(rx_rates.ma_jumps)
    num_species = size(u,1) #NOTE assumes u is a matrix with ith column being the ith site
    num_sites = DiffEqJump.num_sites(spatial_system)

    pqdata = Vector{typeof(t)}(undef, num_sites)
    for site in 1:num_sites
        update_reaction_rates!(rx_rates, 1:num_majumps, u, site)
        update_hop_rates!(hop_rates, 1:num_species, u, site, spatial_system)
        pqdata[site] = t + randexp(aggregation.rng) / (total_site_rx_rate(rx_rates, site)+total_site_hop_rate(hop_rates, site))
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
        update_site_time!(p, source_site, t)
        update_site_time!(p, target_site, t)
    else
        site = jump.src
        update_rates_after_reaction!(p, u, site, reaction_id_from_jump(p,jump))
        update_site_time!(p, site, t)
    end
end

function update_site_time!(p, site, t)
    @unpack rx_rates, hop_rates, rng, pq = p
    site_rate = (total_site_rx_rate(rx_rates, site)+total_site_hop_rate(hop_rates, site))
    if site_rate > zero(typeof(site_rate))
        update!(pq, site, t + randexp(rng) / site_rate)
    else
        update!(pq, site, typemax(t))
    end
end

######################## helper routines for all spatial SSAs ########################
function update_rates_after_reaction!(p, u, site, reaction_id)
    update_reaction_rates!(p.rx_rates, p.dep_gr[reaction_id], u, site)
    update_hop_rates!(p.hop_rates, p.jumptovars_map[reaction_id], u, site, p.spatial_system)
end

function update_rates_after_hop!(p, u, source_site, target_site, species)
    update_reaction_rates!(p.rx_rates, p.vartojumps_map[species], u, source_site)
    update_hop_rate!(p.hop_rates, species, u, source_site, p.spatial_system)
    
    update_reaction_rates!(p.rx_rates, p.vartojumps_map[species], u, target_site)
    update_hop_rate!(p.hop_rates, species, u, target_site, p.spatial_system)
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
        executerx!((@view integrator.u[:,jump.src]), rx_index, p.rx_rates.ma_jumps)
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
    jump.jidx <= p.numspecies
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
    jump.jidx - p.numspecies
end

"""
number of constant rate jumps
"""
num_constant_rate_jumps(aggregator::NSMJumpAggregation) = 0
