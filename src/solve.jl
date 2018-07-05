function solve{P,recompile_flag}(
  jump_prob::AbstractJumpProblem{P},
  alg::DiffEqBase.DEAlgorithm,timeseries=[],ts=[],ks=[],recompile::Type{Val{recompile_flag}}=Val{true};
  kwargs...)

  integrator = init(jump_prob,alg,timeseries,ts,ks,recompile;kwargs...)
  solve!(integrator)
  integrator.sol
end

function init{P,recompile_flag}(
  jump_prob::AbstractJumpProblem{P},
  alg::DiffEqBase.DEAlgorithm,timeseries=[],ts=[],ks=[],recompile::Type{Val{recompile_flag}}=Val{true};
  callback=nothing,
  kwargs...)

  integrator = init(jump_prob.prob,alg,timeseries,ts,ks,recompile;
                    callback=CallbackSet(callback,jump_prob.jump_callback),
                    kwargs...)
end
