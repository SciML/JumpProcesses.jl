function total_variable_rate(jumps::JumpSet, u, p, t)
    sum_rate = 0.0

    vjumps = jumps.variable_jumps
    if !isempty(vjumps)
        for jump in vjumps
            sum_rate += jump.rate(u, p, t)
        end
    end

    return sum_rate
end


mutable struct GillespieIntegCallbackEventCache
    prev_time::Float64
    prev_threshold::Float64
    current_time::Float64
    current_threshold::Float64  
    cumulative_rate::Float64
    jumps::JumpSet
    function GillespieIntegCallbackEventCache(jumps::JumpSet)
        initial_threshold = -log(rand())
        new(0.0, initial_threshold, 0.0, initial_threshold, 0.0, jumps)
    end
end

# Condition function using 4-point Gaussian quadrature to determine event times
function gillespie_integcallback_jumps_condition(cache::GillespieIntegCallbackEventCache, u, t, integrator)
    if integrator.t != cache.current_time
        cache.prev_threshold = cache.current_threshold
    end
    
    dt = t - cache.prev_time
    if dt == 0.0
        return cache.prev_threshold
    end

    jumps = cache.jumps
    p = integrator.p
    n = 4
    rate_increment = 0.0
    for i in 1:n
        τ = ((dt / 2) * gauss_points[n][i]) + ((t + cache.prev_time) / 2)
        u_τ = integrator(τ)
        total_variable_rate_τ = total_variable_rate(jumps, u_τ, p, τ)
        rate_increment += gauss_weights[n][i] * total_variable_rate_τ
    end
    rate_increment *= (dt / 2)
    
    cache.cumulative_rate += rate_increment
    
    return cache.prev_threshold - rate_increment
end

# Affect function to apply stochastic jumps
function gillespie_integcallback_jumps_affect!(cache::GillespieIntegCallbackEventCache, integrator)
    t = integrator.t
    u = integrator.u
    p = integrator.p
    jumps = cache.jumps

    total_variable_rate_sum = total_variable_rate(jumps, u, p, t)
    if total_variable_rate_sum <= 0
        return
    end

    r = rand() * total_variable_rate_sum
    jump_idx = 0
    prev_rate = 0.0

    vjumps = jumps.variable_jumps
    if !isempty(vjumps)
        for (i, jump) in enumerate(vjumps)
            new_rate = jump.rate(u, p, t)
            prev_rate += new_rate
            if r < prev_rate
                jump_idx = i
                break
            end
        end

        if jump_idx > 0
            vjumps[jump_idx].affect!(integrator)
        else
            error("Jump index $jump_idx out of bounds for available jumps")
        end
    end

    cache.prev_time = t
    cache.prev_threshold = cache.current_threshold
    cache.current_threshold = -log(rand())
    cache.current_time = t
    cache.cumulative_rate = 0.0
end