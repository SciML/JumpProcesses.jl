struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end
struct MatrixFreeTauLeaping <: DiffEqBase.DEAlgorithm end 
struct RegularSSA <: DiffEqBase.DEAlgorithm end

function DiffEqBase.solve(jump_prob::JumpProblem,alg::MatrixFreeTauLeaping,
                            seed = nothing, 
                            dt = error("dt is required for MatrixFreeTauLeaping."))

    # boilerplate from SimpleTauLeaping method
    @assert isempty(jump_prob.jump_callback.continuous_callbacks) # still needs to be a regular jump 
    @assert isempty(jump_prob.jump_callback.discrete_callbacks)  
    prob = jump_prob.prob
    seed === nothing ? rng = Xorshifts.Xoroshiro128Plus() : rng = Xorshifts.Xoroshiro128Plus(seed)

    rj = jump_prob.regular_jump 
    rate = rj.rate # rate function rate(out,u,p,t)
    c = rj.c # matrix-free operator c(u_buffer, uprev, tprev, counts, p)
    u0 = copy(prob.u0) 
    du = similar(u0)
    rate_cache = zeros(eltype(u0), length(u0))

    tspan = prob.tspan 
    p = prob.p 
    # mark = nothing # https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/250
    n = Int((tspan[2] - tspan[1])/dt) + 1 
    u = Vector{typeof(prob.u0)}(undef,n) 
    u[1] = u0 
    t = [tspan[1] + i*dt for i in 0:n-1] 

    # iteration variables 
    counts = zeros(Int,length(u0)) # counts for each variable
      
    for i in 2:n # iterate over dt-slices 
        uprev = u[i-1]
        tprev = t[i-1] 
        rate(rate_cache,uprev,p,tprev) 
        rate_cache .*= dt # multiply by the width of the time interval
        counts .= pois_rand.((rng,), rate_cache) # set counts to the poisson arrivals with our given rates
        c(u[i], uprev, tprev, counts, p)
    end 

    sol = DiffEqBase.build_solution(prob,alg,t,u, 
                                    calculate_error = false,
                                    interp = DiffEqBase.ConstantInterpolation(t,u))
end

function DiffEqBase.solve(jump_prob::JumpProblem,alg::SimpleTauLeaping;
                          seed = nothing,
                          dt = error("dt is required for SimpleTauLeaping"))

  @assert isempty(jump_prob.jump_callback.continuous_callbacks)
  @assert isempty(jump_prob.jump_callback.discrete_callbacks)
  prob = jump_prob.prob 
  seed === nothing ? rng = Xorshifts.Xoroshiro128Plus() :
                    rng = Xorshifts.Xoroshiro128Plus(seed) 

  rj = jump_prob.regular_jump 
  rate = rj.rate 
  c = rj.c 
  dc = zero(rj.dc)
  fill!(dc,0)
  rate_cache = zeros(eltype(prob.u0), size(dc,2)) 

  u0 = copy(prob.u0) 
  tspan = prob.tspan 
  p = prob.p 
  mark = nothing # https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/250

  n = Int((tspan[2] - tspan[1])/dt) + 1 
  u = Vector{typeof(prob.u0)}(undef,n) 
  u[1] = u0 
  t = [tspan[1] + i*dt for i in 0:n-1] 

  counts = zeros(Int,size(dc,2))
  update = similar(u0) 

  rj.constant_c && c(dc,u0,p,tspan[1],mark) 

  for i in 2:n
      uprev = u[i-1]
      tprev = t[i-1] 
      rate(rate_cache,uprev,p,tprev) 
      rate_cache .*= dt 
      counts .= pois_rand.((rng,), rate_cache) # set counts to the poisson arrivals with our given rates
      !rj.constant_c && c(dc,uprev,p,tprev,mark)  
      mul!(update,dc,counts) 
      u[i] = uprev .+ update 
  end

  sol = DiffEqBase.build_solution(prob,alg,t,u, 
                       calculate_error = false,
                       interp = DiffEqBase.ConstantInterpolation(t,u))
end

function DiffEqBase.solve(jump_prob::JumpProblem,alg::RegularSSA)

  @assert isempty(jump_prob.jump_callback.continuous_callbacks)
  @assert isempty(jump_prob.jump_callback.discrete_callbacks)
  prob = jump_prob.prob

  rj = jump_prob.regular_jump
  rate = rj.rate
  c = rj.c
  dc = zero(rj.dc)
  fill!(dc,0)
  rate_cache = zeros(size(dc,2))
  rate_sum = similar(rate_cache)

  u0 = copy(prob.u0)
  tspan = prob.tspan
  p = prob.p
  mark = nothing

  u =[u0]
  t = [tspan[1]]

  rj.constant_c && c(dc,u0,p,tspan[1],mark)

  curt = tspan[1]
  while curt < tspan[2]
      rate(rate_cache,u[end],p,curt)
      cumsum!(rate_sum,rate_cache)
      ttnj = randexp()/rate_sum[end]
      r = rand()
      rate_sum ./= rate_sum[end]
      i = searchsortedfirst(rate_sum,r)
      !rj.constant_c && c(dc,u[end],p,curt,mark)
      # Can instead be dc*[0,0,...,1,...,0,0] if sparse
      # https://github.com/JuliaLang/julia/issues/13438
      unext = u[end] .+ @view dc[:,i]
      curt += ttnj
      push!(t,curt)
      push!(u,unext)
  end

  t[end] = tspan[2]

  sol = DiffEqBase.build_solution(prob,alg,t,u,
                       calculate_error = false,
                       interp = DiffEqBase.ConstantInterpolation(t,u))
end

export SimpleTauLeaping, RegularSSA
