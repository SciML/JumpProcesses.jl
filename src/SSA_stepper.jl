# Integrator specifically for SSA
# Built to have 0-overhead stepping

struct SSAStepper <: DiffEqBase.DEAlgorithm end

mutable struct SSAIntegrator{F,uType,tType,P,S,CB,SA,OPT,TS} <: DiffEqBase.DEIntegrator{SSAStepper,Nothing,uType,tType}
    f::F
    u::uType
    t::tType
    tprev::tType
    p::P
    sol::S
    i::Int
    tstop::tType
    cb::CB
    saveat::SA
    save_everystep::Bool
    save_end::Bool
    cur_saveat::Int
    opts::OPT
    tstops::TS
    tstops_idx::Int
    u_modified::Bool
    keep_stepping::Bool          # false if should terminate a simulation
end

(integrator::SSAIntegrator)(t) = copy(integrator.u)
(integrator::SSAIntegrator)(out,t) = (out .= integrator.u)

function DiffEqBase.__solve(jump_prob::JumpProblem,
                         alg::SSAStepper;
                         kwargs...)
    integrator = init(jump_prob,alg;kwargs...)
    solve!(integrator)
    integrator.sol
end

function DiffEqBase.solve!(integrator)

    end_time = integrator.sol.prob.tspan[2]

    while integrator.keep_stepping && (integrator.t < end_time) # It stops before adding a tstop over
        step!(integrator)
    end

    integrator.t = end_time

    if integrator.saveat !== nothing && !isempty(integrator.saveat)
        # Split to help prediction
        while integrator.cur_saveat <= length(integrator.saveat) &&
           integrator.saveat[integrator.cur_saveat] < integrator.t

            push!(integrator.sol.t,integrator.saveat[integrator.cur_saveat])
            push!(integrator.sol.u,copy(integrator.u))
            integrator.cur_saveat += 1

        end
    end

    if integrator.save_end && integrator.sol.t[end] != end_time
        push!(integrator.sol.t,end_time)
        push!(integrator.sol.u,copy(integrator.u))
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
                         tstops = (),
                         numsteps_hint=100)
    if !(jump_prob.prob isa DiscreteProblem)
        error("SSAStepper only supports DiscreteProblems.")
    end
    @assert isempty(jump_prob.jump_callback.continuous_callbacks)

    if alias_jump
      cb = jump_prob.jump_callback.discrete_callbacks[end]
      if seed !== nothing
          Random.seed!(cb.condition.rng,seed)
      end
    else
      cb = deepcopy(jump_prob.jump_callback.discrete_callbacks[end])
      if seed === nothing
          Random.seed!(cb.condition.rng,seed_multiplier()*rand(UInt64))
      else
          Random.seed!(cb.condition.rng,seed)
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


    sol = DiffEqBase.build_solution(prob,alg,t,u,dense=false,
                         calculate_error = false,
                         destats = DiffEqBase.DEStats(0),
                         interp = DiffEqBase.ConstantInterpolation(t,u))
    save_everystep = any(cb.save_positions)

    if typeof(saveat) <: Number
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
     sizehint!(u,length(_saveat)+1)
     sizehint!(t,length(_saveat)+1)
   elseif save_everystep
     sizehint!(u,numsteps_hint)
     sizehint!(t,numsteps_hint)
   else
     sizehint!(u,2)
     sizehint!(t,2)
   end

    integrator = SSAIntegrator(prob.f,copy(prob.u0),prob.tspan[1],prob.tspan[1],prob.p,
                               sol,1,prob.tspan[1],
                               cb,_saveat,save_everystep,save_end,cur_saveat,
                               opts,tstops,1,false,true)
    cb.initialize(cb,u[1],prob.tspan[1],integrator)
    integrator
end

DiffEqBase.add_tstop!(integrator::SSAIntegrator,tstop) = integrator.tstop = tstop

function DiffEqBase.step!(integrator::SSAIntegrator)
    integrator.tprev = integrator.t

    end_time = integrator.sol.prob.tspan[2]
    next_jump_time = integrator.t >= integrator.tstop ? end_time : integrator.tstop

    doaffect = false
    if !isempty(integrator.tstops) &&
        integrator.tstops_idx <= length(integrator.tstops) &&
        integrator.tstops[integrator.tstops_idx] < next_jump_time

        integrator.t = integrator.tstops[integrator.tstops_idx]
        integrator.tstops_idx += 1
    else
        integrator.t = next_jump_time
        if integrator.t >= end_time
            integrator.t = end_time
            return
        end
        doaffect = true # delay effect until after saveat
    end

    @inbounds if integrator.saveat !== nothing && !isempty(integrator.saveat)
        # Split to help prediction
        while integrator.cur_saveat < length(integrator.saveat) &&
           integrator.saveat[integrator.cur_saveat] < integrator.t

            saved = true
            push!(integrator.sol.t,integrator.saveat[integrator.cur_saveat])
            push!(integrator.sol.u,copy(integrator.u))
            integrator.cur_saveat += 1
        end
    end

    doaffect && integrator.cb.affect!(integrator)

    if !(typeof(integrator.opts.callback.discrete_callbacks)<:Tuple{})
        discrete_modified,saved_in_cb = DiffEqBase.apply_discrete_callback!(integrator,integrator.opts.callback.discrete_callbacks...)
    else
        saved_in_cb = false
    end

    !saved_in_cb && savevalues!(integrator)
    nothing
end

function DiffEqBase.savevalues!(integrator::SSAIntegrator,force=false)
    saved, savedexactly = false, false

    # No saveat in here since it would only use previous values,
    # so in the specific case of SSAStepper it's already handled

    if integrator.save_everystep || force
        saved = true
        savedexactly = true
        push!(integrator.sol.t,integrator.t)
        push!(integrator.sol.u,copy(integrator.u))
    end

    saved, savedexactly
end

function reset_aggregated_jumps!(integrator::SSAIntegrator,uprev = nothing)
     reset_aggregated_jumps!(integrator,uprev,integrator.cb)
     nothing
end

function DiffEqBase.terminate!(integrator::SSAIntegrator, retcode = :Terminated)
    integrator.keep_stepping = false
    integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol, retcode)
    nothing
end

export SSAStepper


