struct SimpleTauLeaping <: DEAlgorithm end

function DiffEqBase.solve(jump_prob::JumpProblem,alg::SimpleTauLeaping;
                          dt = error("dt is required for SimpleTauLeaping"))

  @assert isempty(jump_prob.jump_callback.continuous_callbacks)
  @assert isempty(jump_prob.jump_callback.discrete_callbacks)
  prob = jump_prob.prob

  rj = jump_prob.regular_jump
  rate = rj.rate
  c = rj.c
  dc = rj.c_prototype
  rate_cache = zeros(size(rj.c_prototype,2))

  u0 = copy(prob.u0)
  tspan = prob.tspan
  p = prob.p
  mark = nothing

  n = Int((tspan[2] - tspan[1])/dt) + 1
  u = Vector{typeof(prob.u0)}(n)
  u[1] = u0
  t = [tspan[1] + i*dt for i in 0:n-1]

  counts = zeros(Int,size(rj.c_prototype,2))
  update = similar(u0)

  rj.constant_c && c(dc,u0,p,tspan[1],mark)

  for i in 2:n
      uprev = u[i-1]
      tprev = t[i-1]
      rate(rate_cache,uprev,p,tprev)
      rate_cache .*= dt
      counts .= rand.(Poisson.(rate_cache))
      !rj.constant_c && c(dc,uprev,p,tprev,mark)
      A_mul_B!(update,dc,counts)
      u[i] = uprev .+ update
  end

  sol = build_solution(prob,alg,t,u,
                       calculate_error = false,
                       interp = DiffEqBase.ConstantInterpolation(t,u))
end

export SimpleTauLeaping
