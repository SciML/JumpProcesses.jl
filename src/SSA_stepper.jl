"""
$(TYPEDEF)

Highly efficient integrator for pure jump problems that involve only `ConstantRateJump`s,
`MassActionJump`s, and/or `VariableRateJump`s *with rate bounds*.

## Notes
- Only works with `JumpProblem`s defined from `DiscreteProblem`s.
- Only works with collections of `ConstantRateJump`s, `MassActionJump`s, and
  `VariableRateJump`s with rate bounds.
- Only supports `DiscreteCallback`s for events, which are checked after every step taken by
  `SSAStepper`.
- Only supports a limited subset of the output controls from the common solver interface,
  specifically `save_start`, `save_end`, and `saveat`.
- As when using jumps with ODEs and SDEs, saving controls for whether to save each time a
  jump occurs are via the `save_positions` keyword argument to `JumpProblem`. Note that when
  choosing `SSAStepper` as the timestepper, `save_positions = (true,true)`, `(true,false)`,
  or `(false,true)` are all equivalent. `SSAStepper` will save only the post-jump state in
  the solution object in each of these cases. This is because solution objects generated via
  `SSAStepper` use piecewise-constant interpolation, and can therefore exactly reconstruct
  the sampled jump process path with knowing just the post-jump state. That is, `sol(t)`
  for any `0 <= t <= tstop` will give the exact value of the sampled solution path at `t`
  provided at least one component of `save_positions` is `true`.

## Examples
SIR model:
```julia
using JumpProcesses
β = 0.1 / 1000.0; ν = .01;
p = (β,ν)
rate1(u,p,t) = p[1]*u[1]*u[2]  # β*S*I
function affect1!(integrator)
  integrator.u[1] -= 1         # S -> S - 1
  integrator.u[2] += 1         # I -> I + 1
end
jump = ConstantRateJump(rate1,affect1!)

rate2(u,p,t) = p[2]*u[2]      # ν*I
function affect2!(integrator)
  integrator.u[2] -= 1        # I -> I - 1
  integrator.u[3] += 1        # R -> R + 1
end
jump2 = ConstantRateJump(rate2,affect2!)
u₀    = [999,1,0]
tspan = (0.0,250.0)
prob = DiscreteProblem(u₀, tspan, p)
jump_prob = JumpProblem(prob, Direct(), jump, jump2)
sol = solve(jump_prob, SSAStepper())
```
see the
[tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/)
for details.
"""
struct SSAStepper <: DiffEqBase.DEAlgorithm end

"""
$(TYPEDEF)

Solution objects for pure jump problems solved via `SSAStepper`.

## Fields

$(FIELDS)
"""
mutable struct SSAIntegrator{F, uType, tType, tdirType, P, S, CB, SA, OPT, TS} <:
               AbstractSSAIntegrator{SSAStepper, Nothing, uType, tType}
    """The underlying `prob.f` function. Not currently used."""
    f::F
    """The current solution values."""
    u::uType
    """The current solution time."""
    t::tType
    """The previous time a jump occurred."""
    tprev::tType
    """The direction time is changing in (must be positive, indicating time is increasing)"""
    tdir::tdirType
    """The current parameters."""
    p::P
    """The current solution object."""
    sol::S
    i::Int
    """The next jump time."""
    tstop::tType
    """The jump aggregator callback."""
    cb::CB
    """Times to save the solution at."""
    saveat::SA
    """Whether to save every time a jump occurs."""
    save_everystep::Bool
    """Whether to save at the final step."""
    save_end::Bool
    """Index of the next `saveat` time."""
    cur_saveat::Int
    """Tuple storing callbacks."""
    opts::OPT
    """User supplied times to step to, useful with callbacks."""
    tstops::TS
    tstops_idx::Int
    u_modified::Bool
    keep_stepping::Bool          # false if should terminate a simulation
end

(integrator::SSAIntegrator)(t) = copy(integrator.u)
(integrator::SSAIntegrator)(out, t) = (out .= integrator.u)

function DiffEqBase.get_tstops(integrator::SSAIntegrator)
    @view integrator.tstops[(integrator.tstops_idx):end]
end
DiffEqBase.get_tstops_array(integrator::SSAIntegrator) = DiffEqBase.get_tstops(integrator)

# ODE integrators seem to add tf into tstops which SSAIntegrator does not do
# so must account for it here
function DiffEqBase.get_tstops_max(integrator::SSAIntegrator)
    tstops = DiffEqBase.get_tstops_array(integrator)
    tf = integrator.sol.prob.tspan[2]
    if !isempty(tstops)
        return max(maximum(tstops), tf)
    else
        return tf
    end
end

function DiffEqBase.u_modified!(integrator::SSAIntegrator, bool::Bool)
    integrator.u_modified = bool
end

function DiffEqBase.__solve(jump_prob::JumpProblem, alg::SSAStepper; kwargs...)
    integrator = init(jump_prob, alg; kwargs...)
    solve!(integrator)
    integrator.sol
end

function DiffEqBase.solve!(integrator::SSAIntegrator)
    end_time = integrator.sol.prob.tspan[2]
    while should_continue_solve(integrator) # It stops before adding a tstop over
        step!(integrator)
    end
    integrator.t = end_time

    if integrator.saveat !== nothing && !isempty(integrator.saveat)
        # Split to help prediction
        while integrator.cur_saveat <= length(integrator.saveat) &&
            integrator.saveat[integrator.cur_saveat] < integrator.t
            push!(integrator.sol.t, integrator.saveat[integrator.cur_saveat])
            push!(integrator.sol.u, copy(integrator.u))
            integrator.cur_saveat += 1
        end
    end

    if integrator.save_end && integrator.sol.t[end] != end_time
        push!(integrator.sol.t, end_time)
        push!(integrator.sol.u, copy(integrator.u))
    end

    DiffEqBase.finalize!(integrator.opts.callback, integrator.u, integrator.t, integrator)

    if integrator.sol.retcode === ReturnCode.Default
        integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, ReturnCode.Success)
    end
end

function DiffEqBase.__init(jump_prob::JumpProblem,
        alg::SSAStepper;
        save_start = true,
        save_end = true,
        seed = nothing,
        alias_jump = Threads.threadid() == 1,
        saveat = nothing,
        callback = nothing,
        tstops = eltype(jump_prob.prob.tspan)[],
        numsteps_hint = 100)
    if !(jump_prob.prob isa DiscreteProblem)
        error("SSAStepper only supports DiscreteProblems.")
    end
    @assert isempty(jump_prob.jump_callback.continuous_callbacks)
    if alias_jump
        cb = jump_prob.jump_callback.discrete_callbacks[end]
        if seed !== nothing
            Random.seed!(cb.condition.rng, seed)
        end
    else
        cb = deepcopy(jump_prob.jump_callback.discrete_callbacks[end])
        if seed === nothing
            Random.seed!(cb.condition.rng, rand(UInt64))
        else
            Random.seed!(cb.condition.rng, seed)
        end
    end

    opts = (callback = CallbackSet(callback),)
    prob = jump_prob.prob

    if save_start
        t = [prob.tspan[1]]
        u = [copy(prob.u0)]
    else
        t = typeof(prob.tspan[1])[]
        u = typeof(prob.u0)[]
    end

    save_everystep = any(cb.save_positions)
    sol = DiffEqBase.build_solution(prob, alg, t, u, dense = save_everystep,
        calculate_error = false,
        stats = DiffEqBase.Stats(0),
        interp = DiffEqBase.ConstantInterpolation(t, u))

    if saveat isa Number
        _saveat = prob.tspan[1]:saveat:prob.tspan[2]
    else
        _saveat = saveat
    end

    if _saveat !== nothing && !isempty(_saveat) && _saveat[1] == prob.tspan[1]
        cur_saveat = 2
    else
        cur_saveat = 1
    end

    if _saveat !== nothing && !isempty(_saveat)
        sizehint!(u, length(_saveat) + 1)
        sizehint!(t, length(_saveat) + 1)
    elseif save_everystep
        sizehint!(u, numsteps_hint)
        sizehint!(t, numsteps_hint)
    else
        sizehint!(u, save_start + save_end)
        sizehint!(t, save_start + save_end)
    end

    tdir = sign(prob.tspan[2] - prob.tspan[1])
    (tdir <= 0) &&
        error("The time interval to solve over is non-increasing, i.e. tspan[2] <= tspan[1]. This is not allowed for pure jump problem.")

    integrator = SSAIntegrator(prob.f, copy(prob.u0), prob.tspan[1], prob.tspan[1], tdir,
        prob.p, sol, 1, prob.tspan[1], cb, _saveat, save_everystep,
        save_end, cur_saveat, opts, tstops, 1, false, true)
    cb.initialize(cb, integrator.u, prob.tspan[1], integrator)
    DiffEqBase.initialize!(opts.callback, integrator.u, prob.tspan[1], integrator)
    integrator
end

function DiffEqBase.add_tstop!(integrator::SSAIntegrator, tstop)
    if tstop > integrator.t
        future_tstops = @view integrator.tstops[(integrator.tstops_idx):end]
        insert_index = integrator.tstops_idx + searchsortedfirst(future_tstops, tstop) - 1
        @show insert_index
        Base.insert!(integrator.tstops, insert_index, tstop)
    end
    @show integrator.t,tstop,integrator.tstops
end

# The Jump aggregators should not register the next jump through add_tstop! for SSAIntegrator
# such that we can achieve maximum performance
@inline function register_next_jump_time!(integrator::SSAIntegrator,
        p::AbstractSSAJumpAggregator, t)
    integrator.tstop = p.next_jump_time
    nothing
end

function DiffEqBase.step!(integrator::SSAIntegrator)
    integrator.tprev = integrator.t
    next_jump_time = integrator.tstop > integrator.t ? integrator.tstop :
                     typemax(integrator.tstop)

    doaffect = false
    if !isempty(integrator.tstops) &&
       integrator.tstops_idx <= length(integrator.tstops) &&
       integrator.tstops[integrator.tstops_idx] < next_jump_time
        integrator.t = integrator.tstops[integrator.tstops_idx]
        integrator.tstops_idx += 1
    else
        integrator.t = integrator.tstop
        doaffect = true # delay effect until after saveat
    end

    @inbounds if integrator.saveat !== nothing && !isempty(integrator.saveat)
        # Split to help prediction
        while integrator.cur_saveat < length(integrator.saveat) &&
            integrator.saveat[integrator.cur_saveat] < integrator.t
            saved = true
            push!(integrator.sol.t, integrator.saveat[integrator.cur_saveat])
            push!(integrator.sol.u, copy(integrator.u))
            integrator.cur_saveat += 1
        end
    end

    # FP error means the new time may equal the old if the next jump time is
    # sufficiently small, hence we add this check to execute jumps until
    # this is no longer true.
    integrator.u_modified = true
    while integrator.t == integrator.tstop
        doaffect && integrator.cb.affect!(integrator)
    end

    jump_modified_u = integrator.u_modified

    if !(integrator.opts.callback.discrete_callbacks isa Tuple{})
        discrete_modified, saved_in_cb = DiffEqBase.apply_discrete_callback!(integrator,
            integrator.opts.callback.discrete_callbacks...)
    else
        saved_in_cb = false
    end

    !saved_in_cb && jump_modified_u && savevalues!(integrator)

    nothing
end

function DiffEqBase.savevalues!(integrator::SSAIntegrator, force = false)
    saved, savedexactly = false, false

    # No saveat in here since it would only use previous values,
    # so in the specific case of SSAStepper it's already handled

    if integrator.save_everystep || force
        saved = true
        savedexactly = true
        push!(integrator.sol.t, integrator.t)
        push!(integrator.sol.u, copy(integrator.u))
    end

    saved, savedexactly
end

function should_continue_solve(integrator::SSAIntegrator)
    end_time = integrator.sol.prob.tspan[2]

    # we continue the solve if there is a tstop between now and end_time
    has_tstop = !isempty(integrator.tstops) &&
                integrator.tstops_idx <= length(integrator.tstops) &&
                integrator.tstops[integrator.tstops_idx] < end_time

    # we continue the solve if there will be a jump between now and end_time
    has_jump = integrator.t < integrator.tstop < end_time

    integrator.keep_stepping && (has_jump || has_tstop)
end

function reset_aggregated_jumps!(integrator::SSAIntegrator, uprev = nothing)
    reset_aggregated_jumps!(integrator, uprev, integrator.cb)
    nothing
end

function DiffEqBase.terminate!(integrator::SSAIntegrator, retcode = ReturnCode.Terminated)
    integrator.keep_stepping = false
    integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, retcode)
    nothing
end

export SSAStepper

function SciMLBase.isdenseplot(sol::ODESolution{
        T, N, uType, uType2, DType, tType, rateType, discType, P,
        SSAStepper}) where {T, N, uType, uType2, DType, tType, rateType, discType, P}
    sol.dense
end
