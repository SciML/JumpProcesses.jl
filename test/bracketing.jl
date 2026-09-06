using JumpProcesses, Test
const JP = JumpProcesses

fluctuation_rate = 0.1
threshold = 25
Δu = 4
bd = BracketData(fluctuation_rate, threshold, Δu)

### Getters ###
species_index = 1
# The fluctuation rate δ corresponds to species brackets (1-δ)*u, (1+δ)*u. So 0 < δ < 1. 
@test 0 < JP.getfr(bd, species_index) < 1
# If u < threshold, then the brackets are (max(u-Δu, 0), u+Δu). So 0 <= threshold and 0 <= Δu.
@test 0 <= JP.gettv(bd, species_index)
@test 0 <= JP.getΔu(bd, species_index)

### Species brackets ###
u = [3, 20, 100] # species vector
species_index = 1
@test JP.get_spec_brackets(bd, species_index, u)[1] == 0
@test JP.get_spec_brackets(bd, species_index, u)[2] == u[1] + Δu
species_index = 2
@test JP.get_spec_brackets(bd, species_index, u)[1] == u[2] - Δu
@test JP.get_spec_brackets(bd, species_index, u)[2] == u[2] + Δu
species_index = 3
@test JP.get_spec_brackets(bd, species_index, u)[1]≈u[3] * (1 - fluctuation_rate) atol=1
@test JP.get_spec_brackets(bd, species_index, u)[2]≈u[3] * (1 + fluctuation_rate) atol=1

### Reaction rate brackets ###
ulow = [2]
uhigh = [10]

# massaction
majump_rates = [0.1] # death at rate 0.1
reactstoch = [[1 => 1]]
netstoch = [[1 => -1]]
majump = MassActionJump(majump_rates, reactstoch,
    netstoch)
reaction_index = 1
@test JP.get_majump_brackets(ulow, uhigh, reaction_index, majump)[1] == majump_rates[1] * ulow[1] # low
@test JP.get_majump_brackets(ulow, uhigh, reaction_index, majump)[2] == majump_rates[1] * uhigh[1] # high

# constant rate
rate(u, params, t) = 1 / u[1]
affect!(integrator) = nothing
crj = ConstantRateJump(rate, affect!)
params = nothing
t = 0.0
@test JP.cjump_brackets(crj, ulow, uhigh, params, t)[1] == 1 / 10 # low
@test JP.cjump_brackets(crj, ulow, uhigh, params, t)[2] == 1 / 2 # high

### Aggregator ###
mutable struct DummyAggregator{T, M, R, CB, BD} <:
               JP.AbstractSSAJumpAggregator{T, M, R, Nothing, Nothing}
    ulow::Vector{Int}
    uhigh::Vector{Int}
    cur_rate_low::Vector{T}
    cur_rate_high::Vector{T}
    sum_rate::T
    ma_jumps::M
    rates::R
    brackets::CB
    bracket_data::BD
end
# one massaction jump, one constant rate jump
cur_rate_low = [0.0, 0.0]
cur_rate_high = [0.0, 0.0]
sum_rate = 0.0
brackets = JP.get_jump_bracket_fwrappers(ulow, params, t, (crj,))
p = DummyAggregator([0], [0], cur_rate_low, cur_rate_high, sum_rate, majump,
    [rate], brackets, bd)

u = [100]
JP.update_u_brackets!(p, u)
@test p.ulow[1]≈u[1] * (1 - fluctuation_rate) atol=1
@test p.uhigh[1]≈u[1] * (1 + fluctuation_rate) atol=1

reaction_index = 1
@test JP.get_jump_brackets(reaction_index, p, params, t)[1] == majump_rates[1] * p.ulow[1]
@test JP.get_jump_brackets(reaction_index, p, params, t)[2] == majump_rates[1] * p.uhigh[1]
reaction_index = 2
@test JP.get_jump_brackets(reaction_index, p, params, t)[1] == rate(p.uhigh, params, t)
@test JP.get_jump_brackets(reaction_index, p, params, t)[2] == rate(p.ulow, params, t)

p = DummyAggregator([0], [0], cur_rate_low, cur_rate_high, sum_rate, majump,
    [rate], brackets, bd)
JP.set_bracketing!(p, u, params, t)
@test p.ulow[1]≈u[1] * (1 - fluctuation_rate) atol=1
@test p.uhigh[1]≈u[1] * (1 + fluctuation_rate) atol=1
@test p.cur_rate_low[1]≈majump_rates[1] * u[1] * (1 - fluctuation_rate) atol=1
@test p.cur_rate_high[1]≈majump_rates[1] * u[1] * (1 + fluctuation_rate) atol=1
@test p.cur_rate_low[2] == rate(p.uhigh, params, t)
@test p.cur_rate_high[2] == rate(p.ulow, params, t)
@test p.sum_rate ≈ sum(p.cur_rate_high)

# non-monotonic rate function
bounded_crj = ConstantRateJump((u, p, t) -> u[1] / (1 + u[2]), affect!;
    lrate = (ulow, uhigh, p, t) -> ulow[1] / (1 + uhigh[2]),
    urate = (ulow, uhigh, p, t) -> uhigh[1] / (1 + ulow[2]))

box_low, box_high = [2, 5], [6, 9]
box_states = [[a, b] for a in box_low[1]:box_high[1], b in box_low[2]:box_high[2]]

lo = JP.lower_rate_bound(bounded_crj, box_low, box_high, params, t)
hi = JP.upper_rate_bound(bounded_crj, box_low, box_high, params, t)
@test all(lo <= bounded_crj.rate(u, params, t) <= hi for u in box_states)
