"""
A file with helper functions for spatial simulations
"""

"""
stores info for a spatial jump
"""
struct SpatialJump{J}
    "source location"
    src::J

    "index of jump as a hop or reaction"
    jidx::Int

    "destination location, equal to src for within-site reactions"
    dst::J
end

function Base.show(io::IO, ::MIME"text/plain", jump::SpatialJump)
    println(io,
        "SpatialJump with source $(jump.src), destination $(jump.dst) and index $(jump.jidx).")
end

######################## helper routines for all spatial SSAs ########################
"""
    sample_jump_direct(p, site)

sample jump at site with direct method
"""
sample_jump_direct(p, site) = sample_jump_direct(p.rx_rates, p.hop_rates, site, p.spatial_system, p.rng)

function sample_jump_direct(rx_rates, hop_rates, site, spatial_system, rng)
    numspecies = size(hop_rates.rates, 1)
    if rand(rng) * (total_site_rate(rx_rates, hop_rates, site)) <
       total_site_rx_rate(rx_rates, site)
        rx = sample_rx_at_site(rx_rates, site, rng)
        return SpatialJump(site, rx + numspecies, site)
    else
        species_to_diffuse, target_site = sample_hop_at_site(hop_rates, site, rng,
                                                             spatial_system)
        return SpatialJump(site, species_to_diffuse, target_site)
    end
end

function total_site_rate(rx_rates::RxRates, hop_rates::AbstractHopRates, site)
    total_site_hop_rate(hop_rates, site) + total_site_rx_rate(rx_rates, site)
end

function update_rates_after_reaction!(p, integrator, site, reaction_id)
    u = integrator.u
    update_rx_rates!(p.rx_rates, p.dep_gr[reaction_id], integrator, site)
    update_hop_rates!(p.hop_rates, p.jumptovars_map[reaction_id], u, site, p.spatial_system)
end

function update_rates_after_hop!(p, integrator, source_site, target_site, species)
    u = integrator.u
    update_rx_rates!(p.rx_rates, p.vartojumps_map[species], integrator, source_site)
    update_hop_rates!(p.hop_rates, species, u, source_site, p.spatial_system)

    update_rx_rates!(p.rx_rates, p.vartojumps_map[species], integrator, target_site)
    update_hop_rates!(p.hop_rates, species, u, target_site, p.spatial_system)
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
        rx_index = reaction_id_from_jump(p, jump)
        @inbounds executerx!((@view integrator.u[:, jump.src]), rx_index,
                             get_majumps(p.rx_rates))
    end
    # save jump that was just executed
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
    @inbounds integrator.u[species, source_site] -= 1
    @inbounds integrator.u[species, target_site] += 1
end

"""
    reaction_id_from_jump(p,jump)

return reaction id by subtracting the number of hops
"""
function reaction_id_from_jump(p, jump)
    jump.jidx - p.numspecies
end
