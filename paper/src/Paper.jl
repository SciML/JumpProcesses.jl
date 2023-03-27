module Paper

using JumpProcesses,
      Graphs, OrdinaryDiffEq, Plots, Statistics, PiecewiseDeterministicMarkovProcesses
import Distributions: Exponential
import LinearAlgebra: I
const PDMP = PiecewiseDeterministicMarkovProcesses

using PyCall

struct PyTick end
struct PDMPCHV end

include("utils.jl")
include("viz.jl")
include("hawkes.jl")

export PyTick, PDMPCHV
export reset_history!, histories, conditional_rate, empirical_rate, qq
export half_page, pgfkw
export hawkes_rate, hawkes_rate_closure, hawkes_jump, hawkes_Λ, hawkes_problem

end
