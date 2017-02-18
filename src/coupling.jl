SplitCoupledJumpProblem(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem,aggregator::AbstractAggregatorAlgorithm,coupling_map::Vector{Tuple{Int64,Int64}})= JumpProblem(cat_problems(prob.prob,prob_control.prob),aggregator,build_split_jumps(prob,prob_control,coupling_map)...)

# make new problem by joining initial_data
function cat_problems(prob::DiscreteProblem,prob_control::DiscreteProblem)
  u0_coupled = CoupledArray(prob.u0,prob_control.u0,true)
  DiscreteProblem(u0_coupled,prob.tspan)
end

function cat_problems(prob::AbstractODEProblem,prob_control::AbstractODEProblem)
  l = length(prob.u0) # add l_c = length(prob_control.u0)
  new_f = function (t,u,du)
    prob.f(t,u.u,@view du[1:l])
    prob_control.f(t,u.u_control,@view du[l+1:2*l])
  end
  u0_coupled = CoupledArray(prob.u0,prob_control.u0,true)
  ODEProblem(new_f,u0_coupled,prob.tspan)
end

function cat_problems(prob::DiscreteProblem,prob_control::AbstractODEProblem)
  l = length(prob.u0) # add l_c = length(prob_control.u0)
  if !(typeof(prob.f) <: typeof(DiffEqBase.DISCRETE_INPLACE_DEFAULT))
    warn("Coupling to DiscreteProblem with nontrivial f.")
  end
  new_f = function (t,u,du)
    prob.f(t,u.u,@view du[1:l])
    prob_control.f(t,u.u_control,@view du[l+1:2*l])
  end
  u0_coupled = CoupledArray(prob.u0,prob_control.u0,true)
  ODEProblem(new_f,u0_coupled,prob.tspan)
end


function cat_problems(prob::AbstractSDEProblem,prob_control::AbstractSDEProblem)
  l = length(prob.u0)
  new_f = function (t,u,du)
    prob.f(t,u.u,@view du[1:l])
    prob_control.f(t,u.u_control,@view du[l+1:2*l])
  end
  new_g = function (t,u,du)
    prob.g(t,u.u,@view du[1:l])
    prob_control.g(t,u.u_control,@view du[l+1:2*l])
  end
  u0_coupled = CoupledArray(prob.u0,prob_control.u0,true)
  SDEProblem(new_f,new_g,u0_coupled,prob.tspan)
end

function cat_problems(prob::AbstractSDEProblem,prob_control::AbstractODEProblem)
  l = length(prob.u0)
  new_f = function (t,u,du)
    prob.f(t,u.u,@view du[1:l])
    prob_control.f(t,u.u_control,@view du[l+1:2*l])
  end
  new_g = function (t,u,du)
    prob.g(t,u.u,@view du[1:l])
    for i in l+1:2*l
      du[i] = 0.
    end
  end
  u0_coupled = CoupledArray(prob.u0,prob_control.u0,true)
  SDEProblem(new_f,new_g,u0_coupled,prob.tspan)
end

function cat_problems(prob::AbstractSDEProblem,prob_control::DiscreteProblem)
  l = length(prob.u0)
  if !(typeof(prob_control.f) <: typeof(DiffEqBase.DISCRETE_INPLACE_DEFAULT))
    warn("Coupling to DiscreteProblem with nontrivial f.")
  end
  new_f = function (t,u,du)
    prob.f(t,u.u,@view du[1:l])
    prob_control.f(t,u.u_control,@view du[l+1:2*l])
  end
  new_g = function (t,u,du)
    prob.g(t,u.u,@view du[1:l])
    for i in l+1:2*l
      du[i] = 0.
    end
  end
  u0_coupled = CoupledArray(prob.u0,prob_control.u0,true)
  SDEProblem(new_f,new_g,u0_coupled,prob.tspan)
end

cat_problems(prob_control::AbstractODEProblem,prob::DiscreteProblem) = cat_problems(prob,prob_control)
cat_problems(prob_control::DiscreteProblem,prob::AbstractSDEProblem) = cat_problems(prob,prob_control)
cat_problems(prob_control::AbstractODEProblem,prob::AbstractSDEProblem) = cat_problems(prob,prob_control)


# this only depends on the jumps in prob, not prob.prob
function build_split_jumps(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem,coupling_map::Vector{Tuple{Int64,Int64}})
  num_jumps = length(prob.discrete_jump_aggregation.rates)
  num_jumps_control = length(prob_control.discrete_jump_aggregation.rates)
  jumps = []
  # overallocates, will fix later
  uncoupled = deleteat!(Vector(1:num_jumps),[c[1] for c in coupling_map ])
  uncoupled_control = deleteat!(Vector(1:num_jumps_control),[c[2] for c in coupling_map ])
  for c in uncoupled   # make uncoupled jumps in prob
    new_rate = prob.discrete_jump_aggregation.rates[c]
    new_affect! = prob.discrete_jump_aggregation.affects![c]
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
  end
  for c in uncoupled_control  # make uncoupled jumps in prob_control
    rate = prob_control.discrete_jump_aggregation.rates[c]
    new_rate = (t,u)->rate(t,u.u_control)
    affect! = prob_control.discrete_jump_aggregation.affects![c]
    new_affect! = function (integrator)
        flip_u!(integrator.u)
        affect!(integrator)
        flip_u!(integrator.u)
    end
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
  end

  for c in coupling_map # make coupled jumps. 3 new jumps for each old one
    rate = prob.discrete_jump_aggregation.rates[c[1]]
    rate_control = prob_control.discrete_jump_aggregation.rates[c[2]]
    affect! = prob.discrete_jump_aggregation.affects![c[1]]
    affect_control! = prob_control.discrete_jump_aggregation.affects![c[2]]
    # shared jump
    new_affect! = function (integrator)
        affect!(integrator)
        flip_u!(integrator.u)
        affect_control!(integrator)
        flip_u!(integrator.u)
    end
    new_rate = (t,u)->min(rate(t,u.u),rate_control(t,u.u_control))
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
     # only prob
    new_affect! = affect!
    new_rate = (t,u)->rate(t,u.u)-min(rate(t,u.u),rate_control(t,u.u_control))
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
    # only prob_control
    new_affect! = function (integrator)
        flip_u!(integrator.u)
        affect!(integrator)
        flip_u!(integrator.u)
    end
    new_rate = (t,u)->rate_control(t,u.u_control)-min(rate(t,u.u),rate_control(t,u.u_control))
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
  end
  jumps
end
