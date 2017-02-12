SplitCoupledJumpProblem(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem,aggregator::AbstractAggregatorAlgorithm,coupling_map::Vector{Tuple{Int64,Int64}})= JumpProblem(cat_problems(prob,prob_control),aggregator,build_split_jumps(prob,prob_control,coupling_map)...)

type CoupledArray{T,T2} <: AbstractArray{Float64,1}
  u::T
  u_control::T2
  order::Bool
end

Base.length(A::CoupledArray) = length(A.u) + length(A.u_control)
Base.size(A::CoupledArray) = (length(A),)
function Base.getindex(A::CoupledArray,i::Int)
  if A.order == true
    i <= length(A.u) ? A.u[i] : A.u_control[i-length(A.u)]
  else
    i <= length(A.u) ? A.u_control[i] : A.u[i-length(A.u)]
  end
end

function Base.getindex(A::CoupledArray,I...)
  A[sub2ind(A.u,I...)]
end

function Base.getindex(A::CoupledArray,I::CartesianIndex{1})
  A[I[1]]
end

Base.setindex!(A::CoupledArray,v,I...) = (A[sub2ind(A.u,I...)] = v)
Base.setindex!(A::CoupledArray,v,I::CartesianIndex{1}) = (A[I[1]] = v)
function Base.setindex!(A::CoupledArray,v,i::Int)
  if A.order == true
    i <= length(A.u) ? (A.u[i] = v) : (A.u_control[i-length(A.u)] = v)
  else
    i <= length(A.u) ? (A.u_control[i] = v) : (A.u[i-length(A.u)] = v)
  end
end

linearindexing{T<:CoupledArray}(::Type{T}) = Base.LinearFast()
similar(A::CoupledArray) = deepcopy(A)


function recursivecopy!{T<:CoupledArray}(dest::T, src::T)
  recursivecopy!(dest.u,src.u)
  recursivecopy!(dest.u_control,src.u_control)
end

display(A::CoupledArray) = display(A.u)
show(A::CoupledArray) = show(A.u)
plot_indices(A::CoupledArray) = eachindex(A.u)
flip_u!(A::CoupledArray) = (A.order = !A.order)


# make new discrete problem by joining initial_data
function cat_problems(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem)
  u0_coupled = CoupledArray(prob.prob.u0,prob_control.prob.u0,true)
  DiscreteProblem(u0_coupled,prob.prob.tspan)
end

function build_split_jumps(prob::AbstractJumpProblem,prob_control::AbstractJumpProblem,coupling_map::Vector{Tuple{Int64,Int64}})
  num_jumps = length(prob.discrete_jump_aggregation.rates)
  jumps = []
  # overallocates, will fix later
  uncoupled = deleteat!(Vector(1:num_jumps),[c[1] for c in coupling_map ])
  uncoupled_control = deleteat!(Vector(1:num_jumps),[c[2] for c in coupling_map ])
  for c in uncoupled   # make uncoupled jumps in prob
    new_rate = prob.discrete_jump_aggregation.rates[c]
    new_affect! = prob.discrete_jump_aggregation.affects![c]
    append!(jumps,[ConstantRateJump(new_rate,new_affect!)])
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
    new_rate = (t,u)->rate_control(t,u.u)-min(rate(t,u.u),rate_control(t,u.u_control))
    push!(jumps,ConstantRateJump(new_rate,new_affect!))
  end
  jumps
end
