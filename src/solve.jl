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

  prob,initial_stop,jump_callback = build_jump_problem(jump_prob)
  push!(tstops,initial_stop)
  integrator = init(prob,alg,timeseries,ts,ks,recompile;
                    callback=CallbackSet(callback,jump_callback),
                    tstops=tstops,
                    kwargs...)
end

function build_jump_problem{P<:AbstractODEProblem}(jump_prob::AbstractJumpProblem{P})
  t,end_time,u = jump_prob.prob.tspan[1],jump_prob.prob.tspan[2],jump_prob.prob.u0
  jump_callback = DiscreteCallback(CompoundConstantRateJump(
                                    t,u,end_time,jump_prob.jumps.constant_jumps))
  initial_stop = jump_callback.condition.next_jump
  if typeof(jump_prob.jumps.variable_jumps) <: Tuple{}
    new_prob = jump_prob.prob
  end
  new_prob,initial_stop,jump_callback
end
