# site chosen with DirectCR, rx or hop chosen with Direct

############################ DirectCRonDirect ###################################
const MINJUMPRATE = 2.0^exponent(1e-12)

#NOTE state vector u is a matrix. u[i,j] is species i, site j
#NOTE hopping_constants is a matrix. hopping_constants[i,j] is species i, site j
mutable struct DirectCRonDirectJumpAggregation{J,T,RX,HOP,RNG,DEPGR,VJMAP,JVMAP,SS,U<:PriorityTable,W<:Function} <: AbstractSSAJumpAggregator
    next_jump::SpatialJump{J} #some structure to identify the next event: reaction or hop
    prev_jump::SpatialJump{J} #some structure to identify the previous event: reaction or hop
    next_jump_time::T
    end_time::T
    rx_rates::RX
    hop_rates::HOP
    site_rates::Vector{T}
    # rates::F1 #rates for constant-rate jumps
    # affects!::F2 #affects! function determines the effect of constant-rate jumps
    save_positions::Tuple{Bool,Bool}
    rng::RNG
    dep_gr::DEPGR #dep graph is same for each site
    vartojumps_map::VJMAP #vartojumps_map is same for each site
    jumptovars_map::JVMAP #jumptovars_map is same for each site
    spatial_system::SS
    numspecies::Int #number of species
    rt::U
    ratetogroup::W
end

function DirectCRonDirectJumpAggregation(nj::SpatialJump{J}, njt::T, et::T, rx_rates::RX, hop_rates::HOP, site_rates::Vector{T}, sps::Tuple{Bool,Bool}, rng::RNG, spatial_system::SS; num_specs, minrate=convert(T,MINJUMPRATE), vartojumps_map=nothing, jumptovars_map=nothing, dep_graph=nothing, kwargs...) where {J,T,RX,HOP,RNG,SS}

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

    # mapping from jump rate to group id
    minexponent = exponent(minrate)

    # use the largest power of two that is <= the passed in minrate
    minrate = 2.0^minexponent
    ratetogroup = rate -> priortogid(rate, minexponent)

    # construct an empty initial priority table -- we'll reset this in init
    rt = PriorityTable(ratetogroup, zeros(T, num_sites(spatial_system)), minrate, 2*minrate)

    DirectCRonDirectJumpAggregation{J,T,RX,HOP,RNG,typeof(dg),typeof(vtoj_map),typeof(jtov_map),SS,typeof(rt), typeof(ratetogroup)}(nj, nj, njt, et, rx_rates, hop_rates, site_rates, sps, rng, dg, vtoj_map, jtov_map, spatial_system, num_specs, rt, ratetogroup)
end

############################# Required Functions ##############################
# creating the JumpAggregation structure (function wrapper-based constant jumps)
function aggregate(aggregator::DirectCRonDirect, starting_state, p, t, end_time, constant_jumps, ma_jumps, save_positions, rng; hopping_constants, spatial_system, kwargs...)
    num_species = size(starting_state,1)
    majumps = ma_jumps
    if majumps === nothing
        majumps = MassActionJump(Vector{typeof(end_time)}(), Vector{Vector{Pair{Int,Int}}}(), Vector{Vector{Pair{Int,Int}}}())
    end

    next_jump = SpatialJump{Int}(typemax(Int),typemax(Int),typemax(Int)) #a placeholder
    next_jump_time = typemax(typeof(end_time))
    rx_rates = RxRates(num_sites(spatial_system), majumps)
    hop_rates = HopRates(hopping_constants)
    site_rates = zeros(typeof(end_time), num_sites(spatial_system))

    DirectCRonDirectJumpAggregation(next_jump, next_jump_time, end_time, rx_rates, hop_rates, site_rates, save_positions, rng, spatial_system; num_specs = num_species, kwargs...)
end

# set up a new simulation and calculate the first jump / jump time
function initialize!(p::DirectCRonDirectJumpAggregation, integrator, u, params, t)
    fill_rates_and_get_times!(p, u, t)
    generate_jumps!(p, integrator, params, u, t)
    nothing
end

# calculate the next jump / jump time
function generate_jumps!(p::DirectCRonDirectJumpAggregation, integrator, params, u, t)
    @unpack rx_rates, hop_rates, rng, rt = p

    p.next_jump_time  = t + randexp(rng) / rt.gsum
    
    if p.next_jump_time < p.end_time
        site = sample(p.rt, p.site_rates, p.rng)
        if rand(rng)*(total_site_rate(rx_rates, hop_rates, site)) < total_site_rx_rate(rx_rates, site)
            rx = sample_rx_at_site(rx_rates, site, rng)
            p.next_jump = SpatialJump(site, rx+p.numspecies, site)
        else
            species_to_diffuse, target_site = sample_hop_at_site(hop_rates, site, rng, p.spatial_system)
            p.next_jump = SpatialJump(site, species_to_diffuse, target_site)
        end
    end    
    nothing
end

# execute one jump, changing the system state
function execute_jumps!(p::DirectCRonDirectJumpAggregation, integrator, u, params, t)
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
function fill_rates_and_get_times!(aggregation::DirectCRonDirectJumpAggregation, u, t)
    @unpack spatial_system, rx_rates, hop_rates, site_rates, rt, numspecies = aggregation

    reset!(rx_rates)
    reset!(hop_rates)
    site_rates .= zero(typeof(t))

    num_rxs = DiffEqJump.num_rxs(rx_rates)
    num_sites = DiffEqJump.num_sites(spatial_system)

    for site in 1:num_sites
        update_rx_rates!(rx_rates, 1:num_rxs, u, site)
        update_hop_rates!(hop_rates, 1:numspecies, u, site, spatial_system)
        site_rates[site] = total_site_rate(rx_rates, hop_rates, site)
    end
    # setup PriorityTable
    reset!(rt)
    for (pid,priority) in enumerate(site_rates)
        insert!(rt, pid, priority)
    end
    nothing
end

"""
    update_dependent_rates_and_firing_times!(p, u, t)

recalculate jump rates for jumps that depend on the just executed jump (p.prev_jump)
"""
function update_dependent_rates_and_firing_times!(p::DirectCRonDirectJumpAggregation, u, t)
    site_rates = p.site_rates
    jump = p.prev_jump
    if is_hop(p, jump)
        source_site = jump.src
        target_site = jump.dst
        update_rates_after_hop!(p, u, source_site, target_site, jump.jidx)
        
        # update site rates
        oldrate = site_rates[source_site]
        site_rates[source_site] = total_site_rate(p.rx_rates, p.hop_rates, source_site)
        update!(p.rt, source_site, oldrate, site_rates[source_site])
        
        oldrate = site_rates[target_site]
        p.site_rates[target_site] = total_site_rate(p.rx_rates, p.hop_rates, target_site)
        update!(p.rt, target_site, oldrate, site_rates[target_site])
    else
        site = jump.src
        update_rates_after_reaction!(p, u, site, reaction_id_from_jump(p,jump))

        # update site rates
        oldrate = site_rates[site]
        site_rates[site] = total_site_rate(p.rx_rates, p.hop_rates, site)
        update!(p.rt, site, oldrate, site_rates[site])
    end
end

"""
number of constant rate jumps
"""
num_constant_rate_jumps(aggregator::DirectCRonDirectJumpAggregation) = 0
