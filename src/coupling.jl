SplitCoupledJumpProblem(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem,aggregator::AbstractAggregatorAlgorithm,coupling_map::Vector{Tuple{Int64,Int64}})= JumpProblem(cat_problems(prob,prob_control),aggregator,build_split_jumps(prob,prob_control,coupling_map)...)

# make new discrete problem by joining initial_data
function cat_problems(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem)
  coupled_u0 = cat(1,prob.prob.u0,prob_control.prob.u0)
  DiscreteProblem(coupled_u0,prob.prob.tspan)
end

function build_split_jumps(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem,coupling_map::Vector{Tuple{Int64,Int64}})
  num_jumps = length(prob.discrete_jump_aggregation.rates)
  d = length(prob.prob.u0)
  jumps = []
  uncoupled = deleteat!(Vector(1:num_jumps),[c[1] for c in coupling_map ])
  uncoupled_control = deleteat!(Vector(1:num_jumps),[c[2] for c in coupling_map ])
  for c in uncoupled   # make uncoupled jumps in prob
    new_rate = prob.discrete_jump_aggregation.rates[c]
    new_affect! = prob.discrete_jump_aggregation.affects![c]
    append!(jumps,[ConstantRateJump(new_rate,new_affect!)])
  end
  for c in uncoupled_control  # make uncoupled jumps in prob_control
    rate = prob_control.discrete_jump_aggregation.rates[c]
    new_rate = (t,u)->rate(t,u[d+1:2*d])
    affect! = prob_control.discrete_jump_aggregation.affects![c]
    new_affect! = function (integrator)
        flip_u!(integrator)
        affect!(integrator)
        flip_u!(integrator)
    end
    append!(jumps,[ConstantRateJump(new_rate,new_affect!)])
  end

  for c in coupling_map # make coupled jumps. 3 new jumps for each old one
    rate = prob.discrete_jump_aggregation.rates[c[1]]
    rate_control = prob_control.discrete_jump_aggregation.rates[c[2]]
    affect! = prob.discrete_jump_aggregation.affects![c[1]]
    affect_control! = prob_control.discrete_jump_aggregation.affects![c[2]]
    # shared jump
    new_affect! = function (integrator)
        affect!(integrator)
        flip_u!(integrator)
        affect_control!(integrator)
        flip_u!(integrator)
    end
    new_rate = (t,u)->min(rate(t,u),rate_control(t,u[d+1:2*d]))
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
     # only prob
    new_affect! = affect!
    new_rate = (t,u)->rate(t,u)-min(rate(t,u),rate_control(t,u[d+1:2*d]))
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
    # only prob_control
    new_affect! = function (integrator)
        flip_u!(integrator)
        affect!(integrator)
        flip_u!(integrator)
    end
    new_rate = (t,u)->rate_control(t,u[d+1:2*d])-min(rate(t,u),rate_control(t,u[d+1:2*d]))
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
  end
  jumps
end

# flip positions of u in integrator so that affects can be applied
# is there a better way to build new affects from old ones?
flip_u! =  function (integrator)
  d = length(integrator.u)
  halfd = Int(d//2)
  integrator.u[1:halfd],integrator.u[halfd+1:d] = integrator.u[halfd+1:d],integrator.u[1:halfd]
end
