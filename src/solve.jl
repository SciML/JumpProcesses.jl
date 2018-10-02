function solve(
  jump_prob::DiffEqBase.AbstractJumpProblem{P},
  alg::DiffEqBase.DEAlgorithm,timeseries=[],ts=[],ks=[],recompile::Type{Val{recompile_flag}}=Val{true};
  kwargs...) where {P,recompile_flag}

  integrator = init(jump_prob,alg,timeseries,ts,ks,recompile;kwargs...)
  solve!(integrator)
  integrator.sol
end

function init(
  jump_prob::DiffEqBase.AbstractJumpProblem{P},
  alg::DiffEqBase.DEAlgorithm,timeseries=[],ts=[],ks=[],recompile::Type{Val{recompile_flag}}=Val{true};
  callback=nothing,
  kwargs...) where {P,recompile_flag}

  integrator = init(jump_prob.prob,alg,timeseries,ts,ks,recompile;
                    callback=CallbackSet(callback,jump_prob.jump_callback),
                    kwargs...)
end
