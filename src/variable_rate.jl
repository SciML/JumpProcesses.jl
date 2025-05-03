# Define VariableRateAggregator types
abstract type VariableRateAggregator end
struct VRFRMODE <: VariableRateAggregator end
struct VRDirectCB <: VariableRateAggregator end

function configure_jump_problem(prob, vr_aggregator::VRDirectCB, jumps, cvrjs; rng = DEFAULT_RNG)
    new_prob = prob
    cache = VRDirectCBEventCache(jumps, eltype(prob.tspan); rng)
    variable_jump_callback = build_variable_integcallback(cache, CallbackSet(), cvrjs...)
    cont_agg = cvrjs
    return new_prob, variable_jump_callback, cont_agg
end

function configure_jump_problem(prob, vr_aggregator::VRFRMODE, jumps, cvrjs; rng = DEFAULT_RNG)
    new_prob = extend_problem(prob, cvrjs; rng)
    variable_jump_callback = build_variable_callback(CallbackSet(), 0, cvrjs...; rng)
    cont_agg = cvrjs
    return new_prob, variable_jump_callback, cont_agg
end

function total_variable_rate(vjumps::Tuple{Vararg{VariableRateJump}}, u, p, t, cur_rates::AbstractVector=Vector{typeof(t)}(undef, length(vjumps)))
    sum_rate = zero(t)
    if !isempty(vjumps)
        prev_rate = zero(t)
        @inbounds for (i, jump) in enumerate(vjumps)
            new_rate = jump.rate(u, p, t)
            sum_rate = add_fast(new_rate, prev_rate)
            cur_rates[i] = sum_rate
            prev_rate = sum_rate
        end
    end
    return sum_rate
end

mutable struct VRDirectCBEventCache{T, RNG <: AbstractRNG}
    prev_time::T
    prev_threshold::T
    current_time::T
    current_threshold::T  
    total_rate_cache::T
    rng::RNG
    variable_jumps::Tuple{Vararg{VariableRateJump}}
    rate_funcs::Vector{Function}
    affect_funcs::Vector{Function}
    cur_rates::Vector{T}

    function VRDirectCBEventCache(jumps::JumpSet, ::Type{T}; rng = DEFAULT_RNG) where T
        initial_threshold = randexp(rng, T)
        vjumps = jumps.variable_jumps
        rate_funcs = [jump.rate for jump in vjumps]
        affect_funcs = [jump.affect! for jump in vjumps]
        cur_rates = Vector{T}(undef, length(vjumps))
        new{T, typeof(rng)}(zero(T), initial_threshold, zero(T), initial_threshold,
                           zero(T), rng, vjumps, rate_funcs, affect_funcs, cur_rates)
    end
end

# Condition functor defined directly on the cache
function (cache::VRDirectCBEventCache)(u, t, integrator)
    if integrator.t != cache.current_time
        cache.prev_time = cache.current_time
        cache.prev_threshold = cache.current_threshold
        cache.current_time = integrator.t
    end
    
    dt = t - cache.prev_time
    if dt == 0
        return cache.prev_threshold
    end

    vjumps = cache.variable_jumps
    p = integrator.p
    n = 4
    rate_increment = zero(t)
    for i in 1:n
        τ = ((dt / 2) * gauss_points[n][i]) + ((t + cache.prev_time) / 2)
        u_τ = integrator(τ)
        total_variable_rate_τ = total_variable_rate(vjumps, u_τ, p, τ)
        rate_increment += gauss_weights[n][i] * total_variable_rate_τ
    end
    rate_increment *= (dt / 2)
    
    cache.current_threshold = cache.prev_threshold - rate_increment
    
    return cache.current_threshold
end

# Affect functor defined directly on the cache
function (cache::VRDirectCBEventCache)(integrator)
    t = integrator.t
    u = integrator.u
    p = integrator.p
    rng = cache.rng

    cache.total_rate_cache = total_variable_rate(cache.variable_jumps, u, p, t, cache.cur_rates)
    total_variable_rate_sum = cache.total_rate_cache
    if total_variable_rate_sum <= 0
        return
    end

    r = rand(rng) * total_variable_rate_sum
    vjumps = cache.variable_jumps
    if !isempty(vjumps)
        @inbounds jump_idx = searchsortedfirst(cache.cur_rates, r)
        if 1 <= jump_idx <= length(vjumps)
            cache.affect_funcs[jump_idx](integrator)
        else
            error("Jump index $jump_idx out of bounds for available jumps")
        end
    end

    cache.prev_time = t
    cache.current_threshold = randexp(rng)
    cache.prev_threshold = cache.current_threshold
    cache.current_time = t
end

function wrap_jump_in_integcallback(cache::VRDirectCBEventCache, jump)
    condition = function(u, t, integrator)
        cache(u, t, integrator)
    end
    affect! = function(integrator)
        cache(integrator)
        nothing
    end
    new_cb = ContinuousCallback(condition, affect!;
        idxs = jump.idxs,
        rootfind = jump.rootfind,
        interp_points = jump.interp_points,
        save_positions = jump.save_positions,
        abstol = jump.abstol,
        reltol = jump.reltol)
    return new_cb
end

function build_variable_integcallback(cache::VRDirectCBEventCache, cb, jump, jumps...)
    new_cb = wrap_jump_in_integcallback(cache::VRDirectCBEventCache, jump)
    build_variable_integcallback(cache, CallbackSet(cb, new_cb), jumps...)
end

function build_variable_integcallback(cache::VRDirectCBEventCache, cb, jump)
    CallbackSet(cb, wrap_jump_in_integcallback(cache, jump))
end

# extends prob.u0 to an ExtendedJumpArray with Njumps integrated intensity values,
# of type prob.tspan
function extend_u0(prob, Njumps, rng)
    ttype = eltype(prob.tspan)
    u0 = ExtendedJumpArray(prob.u0, [-randexp(rng, ttype) for i in 1:Njumps])
    return u0
end

function extend_problem(prob::DiffEqBase.AbstractDiscreteProblem, jumps; rng = DEFAULT_RNG)
    error("General `VariableRateJump`s require a continuous problem, like an ODE/SDE/DDE/DAE problem. To use a `DiscreteProblem` bounded `VariableRateJump`s must be used. See the JumpProcesses docs.")
end

function extend_problem(prob::DiffEqBase.AbstractODEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, t)
                _f(du.u, u.u, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = ODEFunction{isinplace(prob)}(jump_f; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

function extend_problem(prob::DiffEqBase.AbstractSDEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, p, t)
                _f(du.u, u.u, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, p, t)
                du = ExtendedJumpArray(_f(u.u, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    if prob.noise_rate_prototype === nothing
        jump_g = function (du, u, p, t)
            prob.g(du.u, u.u, p, t)
        end
    else
        jump_g = function (du, u, p, t)
            prob.g(du, u.u, p, t)
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = SDEFunction{isinplace(prob)}(jump_f, jump_g; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, g = jump_g, u0)
end

function extend_problem(prob::DiffEqBase.AbstractDDEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (du::ExtendedJumpArray, u::ExtendedJumpArray, h, p, t)
                _f(du.u, u.u, h, p, t)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (u::ExtendedJumpArray, h, p, t)
                du = ExtendedJumpArray(_f(u.u, h, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = DDEFunction{isinplace(prob)}(jump_f; sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

# Not sure if the DAE one is correct: Should be a residual of sorts
function extend_problem(prob::DiffEqBase.AbstractDAEProblem, jumps; rng = DEFAULT_RNG)
    _f = SciMLBase.unwrapped_f(prob.f)

    if isinplace(prob)
        jump_f = let _f = _f
            function (out, du::ExtendedJumpArray, u::ExtendedJumpArray, h, p, t)
                _f(out, du.u, u.u, h, p, t)
                update_jumps!(out, u, p, t, length(u.u), jumps...)
            end
        end
    else
        jump_f = let _f = _f
            function (du, u::ExtendedJumpArray, h, p, t)
                out = ExtendedJumpArray(_f(du.u, u.u, h, p, t), u.jump_u)
                update_jumps!(du, u, p, t, length(u.u), jumps...)
                return du
            end
        end
    end

    u0 = extend_u0(prob, length(jumps), rng)
    f = DAEFunction{isinplace(prob)}(jump_f, sys = prob.f.sys,
        observed = prob.f.observed)
    remake(prob; f, u0)
end

function wrap_jump_in_callback(idx, jump; rng = DEFAULT_RNG)
    condition = function(u, t, integrator)
        u.jump_u[idx]
    end
    affect! = function(integrator)
        jump.affect!(integrator)
        integrator.u.jump_u[idx] = -randexp(rng, typeof(integrator.t))
        nothing
    end
    new_cb = ContinuousCallback(condition, affect!;
        idxs = jump.idxs,
        rootfind = jump.rootfind,
        interp_points = jump.interp_points,
        save_positions = jump.save_positions,
        abstol = jump.abstol,
        reltol = jump.reltol)
    return new_cb
end

function build_variable_callback(cb, idx, jump, jumps...; rng = DEFAULT_RNG)
    idx += 1
    new_cb = wrap_jump_in_callback(idx, jump; rng)
    build_variable_callback(CallbackSet(cb, new_cb), idx, jumps...; rng = DEFAULT_RNG)
end

function build_variable_callback(cb, idx, jump; rng = DEFAULT_RNG)
    idx += 1
    CallbackSet(cb, wrap_jump_in_callback(idx, jump; rng))
end

@inline function update_jumps!(du, u, p, t, idx, jump)
    idx += 1
    du[idx] = jump.rate(u.u, p, t)
end

@inline function update_jumps!(du, u, p, t, idx, jump, jumps...)
    idx += 1
    du[idx] = jump.rate(u.u, p, t)
    update_jumps!(du, u, p, t, idx, jumps...)
end
