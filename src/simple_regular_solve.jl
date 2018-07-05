struct SimpleTauLeaping <: DiffEqBase.DEAlgorithm end
struct RegularSSA <: DiffEqBase.DEAlgorithm end

function DiffEqBase.solve(jump_prob::JumpProblem,alg::SimpleTauLeaping;
                          seed = nothing,
                          dt = error("dt is required for SimpleTauLeaping"))

  @assert isempty(jump_prob.jump_callback.continuous_callbacks)
  @assert isempty(jump_prob.jump_callback.discrete_callbacks)
  prob = jump_prob.prob
  seed == nothing ? rng = Xorshifts.Xoroshiro128Plus() :
                    rng = Xorshifts.Xoroshiro128Plus(seed)

  rj = jump_prob.regular_jump
  rate = rj.rate
  c = rj.c
  dc = zeros(rj.dc)
  fill!(dc,0)
  rate_cache = zeros(size(dc,2))

  u0 = copy(prob.u0)
  tspan = prob.tspan
  p = prob.p
  mark = nothing

  n = Int((tspan[2] - tspan[1])/dt) + 1
  u = Vector{typeof(prob.u0)}(n)
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
      counts .= pois_rand.(rate_cache,rng)
      !rj.constant_c && c(dc,uprev,p,tprev,mark)
      A_mul_B!(update,dc,counts)
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
  dc = zeros(rj.dc)
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
