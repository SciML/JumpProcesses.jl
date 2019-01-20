# Integrator specifically for SSA
# Built to have 0-overhead stepping

struct SSAStepper end

mutable struct SSAIntegrator{F,uType,tType,P,S,CB,SA} <: DiffEqBase.DEIntegrator
    f::F
    u::uType
    t::tType
    p::P
    sol::S
    i::Int
    tstop::tType
    cb::CB
    saveat::SA
    save_everystep::Bool
    save_end::Bool
    cur_saveat::Int
end

(integrator::SSAIntegrator)(t) = copy(integrator.u)
(integrator::SSAIntegrator)(out,t) = (out .= integrator.u)

function DiffEqBase.solve(jump_prob::JumpProblem,
                         alg::SSAStepper;
                         kwargs...)
    integrator = init(jump_prob,alg;kwargs...)
    solve!(integrator)
    integrator.sol
end

function DiffEqBase.solve!(integrator)
    end_time = integrator.sol.prob.tspan[2]
    while integrator.t < integrator.tstop # It stops before adding a tstop over
        step!(integrator)
    end

    integrator.t = end_time

    if integrator.saveat !== nothing && !isempty(integrator.saveat)
        # Split to help prediction
        while integrator.cur_saveat < length(integrator.saveat) &&
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

function DiffEqBase.init(jump_prob::JumpProblem,
                         alg::SSAStepper;
                         save_start = true,
                         save_end = true,
                         saveat = nothing)
    @assert isempty(jump_prob.jump_callback.continuous_callbacks)
    @assert length(jump_prob.jump_callback.discrete_callbacks) == 1
    cb = jump_prob.jump_callback.discrete_callbacks[1]
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
     sizehint!(u,10000)
     sizehint!(t,10000)
   else
     sizehint!(u,2)
     sizehint!(t,2)
   end

    integrator = SSAIntegrator(prob.f,copy(prob.u0),prob.tspan[1],prob.p,
                               sol,1,prob.tspan[1],
                               cb,_saveat,save_everystep,save_end,cur_saveat)
    cb.initialize(cb,u[1],prob.tspan[1],integrator)

    integrator
end

DiffEqBase.add_tstop!(integrator::SSAIntegrator,tstop) = integrator.tstop = tstop

function DiffEqBase.step!(integrator::SSAIntegrator)
    integrator.t = integrator.tstop
    integrator.cb.affect!(integrator)
    if integrator.save_everystep
        push!(integrator.sol.t,integrator.t)
        push!(integrator.sol.u,copy(integrator.u))
    end
    @inbounds if integrator.saveat !== nothing && !isempty(integrator.saveat)
        # Split to help prediction
        while integrator.cur_saveat < length(integrator.saveat) &&
           integrator.saveat[integrator.cur_saveat] < integrator.t

            push!(integrator.sol.t,integrator.saveat[integrator.cur_saveat])
            push!(integrator.sol.u,copy(integrator.u))
            integrator.cur_saveat += 1

        end
    end
    nothing
end

export SSAStepper
