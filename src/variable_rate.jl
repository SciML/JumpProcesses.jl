"""
$(TYPEDEF)

An abstract type for aggregators that manage the simulation of `VariableRateJump`s in jump processes. 

## Notes
- In hybrid ODE/SDE systems with general `VariableRateJump`s, `integrator.u` may be an 
  `ExtendedJumpArray` for some aggregators.
"""
abstract type VariableRateAggregator end


################################### VRFRMODE ####################################

"""
$(TYPEDEF)

A concrete `VariableRateAggregator` implementing a first-reaction method variant for
simulating `VariableRateJump`s. `VRFRMODE` (Variable Rate First Reaction Method with
Ordinary Differential Equation) uses a user-selected ODE solver to handle integrating each
jump's intensity / propensity. A callback is also used for each jump to determine when its
integrated intensity reaches a level corresponding to a firing time, and to then execute the
affect associated with the jump at that time.  

## Examples
Simulating a birth-death process with `VRFRMODE`:
```julia
using JumpProcesses, OrdinaryDiffEq  
u0 = [1.0]           # Initial population  
p = [10.0, 0.5]      # [birth rate, death rate]  
tspan = (0.0, 10.0)  

# Birth jump: ∅ → X  
birth_rate(u, p, t) = p[1]
birth_affect!(integrator) = (integrator.u[1] += 1; nothing)  
birth_jump = VariableRateJump(birth_rate, birth_affect!)  

# Death jump: X → ∅  
death_rate(u, p, t) = p[2] * u[1]  
death_affect!(integrator) = (integrator.u[1] -= 1; nothing)  
death_jump = VariableRateJump(death_rate, death_affect!)  

# Problem setup  
oprob = ODEProblem((du, u, p, t) -> du .= 0, u0, tspan, p)  
jprob = JumpProblem(oprob, birth_jump, death_jump; vr_aggregator = VRFRMODE())  
sol = solve(jprob, Tsit5())  
```

## Notes
- Specify `VRFRMODE` in a `JumpProblem` via the `vr_aggregator` keyword argument to select
  its use for handling `VariableRateJump`s. 
- While robust, it may be less performant than `VRDirectCB` due to its integration of each
  individual jump's intensity, and use of one continuous callback per jump to handle
  detection of jump times and implementation of state changes from that jump.  
"""
struct VRFRMODE <: VariableRateAggregator end

function configure_jump_problem(prob, vr_aggregator::VRFRMODE, jumps, cvrjs; 
        rng = DEFAULT_RNG)
    new_prob = extend_problem(prob, cvrjs; rng)
    variable_jump_callback = build_variable_callback(CallbackSet(), 0, cvrjs...; rng)
    cont_agg = cvrjs
    return new_prob, variable_jump_callback, cont_agg
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

################################### VRDirectCB ####################################

"""
$(TYPEDEF)

A concrete `VariableRateAggregator` implementing a direct method-based approach for
simulating `VariableRateJump`s. `VRDirectCB` (Variable Rate Direct Callback) efficiently
samples jump times using one continuous callback to integrate the total intensity /
propensity for all `VariableRateJump`s, sample when the next jump occurs, and then sample
which jump occurs at this time. 

## Examples
Simulating a birth-death process with `VRDirectCB` (default):
```julia
using JumpProcesses, OrdinaryDiffEq  
u0 = [1.0]           # Initial population  
p = [10.0, 0.5]      # [birth rate, death rate coefficient]  
tspan = (0.0, 10.0)  

# Birth jump: ∅ → X  
birth_rate(u, p, t) = p[1] 
birth_affect!(integrator) = (integrator.u[1] += 1; nothing)  
birth_jump = VariableRateJump(birth_rate, birth_affect!)  

# Death jump: X → ∅  
death_rate(u, p, t) = p[2] * u[1]
death_affect!(integrator) = (integrator.u[1] -= 1; nothing)  
death_jump = VariableRateJump(death_rate, death_affect!)  

# Problem setup  
oprob = ODEProblem((du, u, p, t) -> du .= 0, u0, tspan, p)  
jprob = JumpProblem(oprob, birth_jump, death_jump; vr_aggregator = VRDirectCB)  
sol = solve(jprob, Tsit5()) 
```

## Notes  
- `VRDirectCB` is expected to generally be more performant than `VRFRMODE`.
"""
struct VRDirectCB <: VariableRateAggregator end

mutable struct VRDirectCBEventCache{T, RNG <: AbstractRNG}
    prev_time::T
    prev_threshold::T
    current_time::T
    current_threshold::T
    total_rate_cache::T
    rng::RNG
    variable_jumps::Tuple{Vararg{VariableRateJump}}
    cur_rates::Vector{T}

    function VRDirectCBEventCache(jumps::JumpSet, ::Type{T}; rng = DEFAULT_RNG) where T
        initial_threshold = randexp(rng, T)
        vjumps = jumps.variable_jumps
        cur_rates = Vector{T}(undef, length(vjumps))
        new{T, typeof(rng)}(zero(T), initial_threshold, zero(T), initial_threshold,
                           zero(T), rng, vjumps, cur_rates)
    end
end

function configure_jump_problem(prob, vr_aggregator::VRDirectCB, jumps, cvrjs; 
        rng = DEFAULT_RNG)
    new_prob = prob
    cache = VRDirectCBEventCache(jumps, eltype(prob.tspan); rng)
    variable_jump_callback = build_variable_integcallback(cache, CallbackSet(), cvrjs...)
    cont_agg = cvrjs
    return new_prob, variable_jump_callback, cont_agg
end

function total_variable_rate(vjumps::Tuple{Vararg{VariableRateJump}}, u, p, t, 
        cur_rates::AbstractVector, idx=1, prev_rate=zero(t))
    if idx > length(cur_rates)
        return prev_rate
    end
    @inbounds begin
        new_rate = vjumps[idx].rate(u, p, t)
        sum_rate = add_fast(new_rate, prev_rate)
        cur_rates[idx] = sum_rate
        return total_variable_rate(vjumps, u, p, t, cur_rates, idx + 1, sum_rate)
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
        total_variable_rate_τ = total_variable_rate(vjumps, u_τ, p, τ, cache.cur_rates)
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
        execute_affect!(vjumps, integrator, jump_idx)
    end

    cache.prev_time = t
    cache.current_threshold = randexp(rng)
    cache.prev_threshold = cache.current_threshold
    cache.current_time = t
    return nothing
end

function execute_affect!(vjumps::Tuple{Vararg{VariableRateJump}}, integrator, idx)
    if !(1 <= idx <= length(vjumps))
        error("Jump index $idx out of bounds for $(length(vjumps)) jumps")
    end
    @inbounds vjumps[idx].affect!(integrator)
end

function wrap_jump_in_integcallback(cache::VRDirectCBEventCache, jump)
    new_cb = ContinuousCallback(cache, cache;
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
