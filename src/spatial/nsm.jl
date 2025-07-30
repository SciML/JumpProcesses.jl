# Implementation of the Next Subvolume Method

############################ NSM ###################################
#NOTE state vector u is a matrix. u[i,j] is species i, site j
mutable struct NSMJumpAggregation{T, S, F1, F2, RNG, J, RX, HOP, DEPGR, VJMAP, JVMAP,
    PQ, SS} <:
               AbstractSSAJumpAggregator{T, S, F1, F2, RNG}
    next_jump::SpatialJump{J} #some structure to identify the next event: reaction or hop
    prev_jump::SpatialJump{J} #some structure to identify the previous event: reaction or hop
    next_jump_time::T
    end_time::T
    rx_rates::RX
    hop_rates::HOP
    rates::F1 # legacy, not used
    affects!::F2 # legacy, not used
    save_positions::Tuple{Bool, Bool}
    rng::RNG
    dep_gr::DEPGR #dep graph is same for each site
    vartojumps_map::VJMAP #vartojumps_map is same for each site
    jumptovars_map::JVMAP #jumptovars_map is same for each site
    pq::PQ
    spatial_system::SS
    numspecies::Int #number of species
end

function NSMJumpAggregation(
        nj::SpatialJump{J}, njt::T, et::T, rx_rates::RX, hop_rates::HOP,
        sps::Tuple{Bool, Bool},
        rng::RNG, spatial_system::SS; num_specs,
        vartojumps_map = nothing, jumptovars_map = nothing,
        dep_graph = nothing, kwargs...) where {J, T, RX, HOP, RNG, SS}

    # a dependency graph is needed
    if dep_graph === nothing
        dg = make_dependency_graph(num_specs, rx_rates.ma_jumps)
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

    NSMJumpAggregation{T, Nothing, Nothing, Nothing, RNG, J, RX, HOP, typeof(dg),
        typeof(vtoj_map), typeof(jtov_map), typeof(pq), SS}(nj, nj, njt, et,
        rx_rates,
        hop_rates,
        nothing,
        nothing,
        sps, rng, dg,
        vtoj_map,
        jtov_map,
        pq,
        spatial_system,
        num_specs)
end

############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::NSM, starting_state, p, t, end_time, constant_jumps,
        ma_jumps, save_positions, rng; hopping_constants, spatial_system,
        kwargs...)
    num_species = size(starting_state, 1)
    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(end_time)}(),
            Vector{Vector{Pair{Int, Int}}}(),
            Vector{Vector{Pair{Int, Int}}}())
    end

    next_jump = SpatialJump{Int}(typemax(Int), typemax(Int), typemax(Int)) #a placeholder
    next_jump_time = typemax(typeof(end_time))
    rx_rates = RxRates(num_sites(spatial_system), majumps)
    hop_rates = HopRates(hopping_constants, spatial_system)

    NSMJumpAggregation(next_jump, next_jump_time, end_time, rx_rates, hop_rates,
        save_positions, rng, spatial_system; num_specs = num_species,
        kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::NSMJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    fill_rates_and_get_times!(p, integrator, t)
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::NSMJumpAggregation, integrator, u, params, t)
    p.next_jump_time, site = top_with_handle(p.pq)
    p.next_jump_time >= p.end_time && return nothing
    p.next_jump = sample_jump_direct(p, site)
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::NSMJumpAggregation, integrator, u, params, t, affects!)
    # execute jump
    update_state!(p, integrator)

    # update current jump rates and times
    update_dependent_rates_and_firing_times!(p, integrator, t)
    nothing
end

######################## SSA specific helper routines ########################
"""
    fill_rates_and_get_times!(aggregation::NSMJumpAggregation, u, t)

reset all structs, reevaluate all rates, recalculate tentative site firing times, and reinit the priority queue
"""
function fill_rates_and_get_times!(aggregation::NSMJumpAggregation, integrator, t)
    @unpack spatial_system, rx_rates, hop_rates, rng = aggregation
    u = integrator.u

    reset!(rx_rates)
    reset!(hop_rates)

    rxs = 1:num_rxs(rx_rates)
    species = 1:(aggregation.numspecies)
    num_nodes = num_sites(spatial_system)

    pqdata = Vector{typeof(t)}(undef, num_nodes)
    @inbounds for site in 1:num_nodes
        update_rx_rates!(rx_rates, rxs, integrator, site)
        update_hop_rates!(hop_rates, species, u, site, spatial_system)
        pqdata[site] = t + randexp(rng) / (total_site_rate(rx_rates, hop_rates, site))
    end

    aggregation.pq = MutableBinaryMinHeap(pqdata)
    nothing
end

"""
    update_dependent_rates_and_firing_times!(p, integrator, t)

recalculate jump rates for jumps that depend on the just executed jump (p.prev_jump)
"""
function update_dependent_rates_and_firing_times!(p::NSMJumpAggregation, integrator, t)
    u = integrator.u
    jump = p.prev_jump
    if is_hop(p, jump)
        source_site = jump.src
        target_site = jump.dst
        update_rates_after_hop!(p, integrator, source_site, target_site, jump.jidx)
        update_site_time!(p, source_site, t)
        update_site_time!(p, target_site, t)
    else
        site = jump.src
        update_rates_after_reaction!(p, integrator, site, reaction_id_from_jump(p, jump))
        update_site_time!(p, site, t)
    end
    nothing
end

"""
    update_site_time!(p::NSMJumpAggregation, site, t)

update the time of site in the priority queue
"""
function update_site_time!(p::NSMJumpAggregation, site, t)
    site_rate = (total_site_rate(p.rx_rates, p.hop_rates, site))
    if site_rate > zero(typeof(site_rate))
        update!(p.pq, site, t + randexp(p.rng) / site_rate)
    else
        update!(p.pq, site, typemax(t))
    end
    nothing
end

"""
    num_constant_rate_jumps(aggregator::NSMJumpAggregation)

number of constant rate jumps
"""
num_constant_rate_jumps(aggregator::NSMJumpAggregation) = 0
