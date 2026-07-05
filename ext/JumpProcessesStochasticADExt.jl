module JumpProcessesStochasticADExt

# BoundedSSA itself lives in JumpProcesses `src` — it is an ordinary uniformization
# (thinning) SSA and does not depend on StochasticAD. This extension supplies the one
# StochasticAD-specific piece: it makes BoundedSSA's discrete accept/channel decisions
# differentiable by overloading the `_bounded_ssa_bernoulli` hook for `StochasticTriple`
# parameters, via StochasticAD's differentiable `rand(::Bernoulli)`. When StochasticAD
# and Distributions are loaded, `solve(jprob, BoundedSSA(; rate_bound))` /
# `bounded_ssa_path` compose with `derivative_estimate`/`stochastic_triple`.

using JumpProcesses
using StochasticAD
using Distributions: Bernoulli

JumpProcesses._bounded_ssa_bernoulli(p::StochasticAD.StochasticTriple) = rand(Bernoulli(p))

end # module
