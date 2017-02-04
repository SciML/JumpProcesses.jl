function solve{P,algType,recompile_flag}(
  jump_prob::AbstractJumpProblem{P},
  alg::algType,timeseries=[],ts=[],ks=[],recompile::Type{Val{recompile_flag}}=Val{true};
  kwargs...)

  integrator = init(jump_prob,alg,timeseries,ts,ks,recompile;kwargs...)
  solve!(integrator)
  integrator.sol
end

function init{P,algType,recompile_flag}(
  jump_prob::AbstractJumpProblem{P},
  alg::algType,timeseries=[],ts=[],ks=[],recompile::Type{Val{recompile_flag}}=Val{true};
  callback=CallbackSet(),tstops = Float64[],
  kwargs...)

  prob,initial_stops,jump_callbacks = build_jump_problem(jump_prob)
  append!(tstops,initial_stops)
  integrator = init(prob,alg,timeseries,ts,ks,recompile;
                    callback=CallbackSet(callback,jump_callbacks),
                    tstops=tstops,
                    kwargs...)
end

function build_jump_problem{P<:AbstractODEProblem}(jump_prob::AbstractJumpProblem{P})
  t,end_time,u = jump_prob.prob.tspan[1],jump_prob.prob.tspan[2],jump_prob.prob.u0
  discrete_jump_callbacks = ((DiscreteCallback(t,u,c,end_time) for c in jump_prob.jumps.constant_jumps)...)
  initial_stops = [cb.condition.next_jump for cb in discrete_jump_callbacks]
  if typeof(jump_prob.jumps.variable_jumps) <: Tuple{}
    new_prob = jump_prob.prob
  end
  jump_callbacks = CallbackSet(discrete_jump_callbacks...)
  new_prob,initial_stops,jump_callbacks
end
