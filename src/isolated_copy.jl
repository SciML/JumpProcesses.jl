# `isolated_copy`: a copy of a `JumpProblem` with its own jump state and everything else
# shared. Each type that holds jump state declares, next to its definition, the fields that
# hold definitions (`shared_fields`, see aggregators/ssajump.jl); the copy rebuilds an object
# through its default constructor from those fields as they are and copies of all the others.

# A new object of the type of `x`, sharing its `shared_fields` and copying its other fields.
# Every type with `shared_fields` keeps its default positional constructor. The generator
# only reads the field names: `shared_fields` is called by the generated code, where it folds
# to a constant, so that declarations made in other packages, after this one loads, are seen.
# (Building the fields with `ntuple` instead allocated more for some types, such as
# `PriorityTable`.)
@generated function copy_working_state(x::T) where {T}
    fields = map(fieldnames(T)) do name
        field = :(getfield(x, $(QuoteNode(name))))
        :($(QuoteNode(name)) in shared_fields(T) ? $field : copy_working($field))
    end
    return :(T($(fields...)))
end

# Copies of the fields that solvers modify. Numbers, `BigFloat`s included, are never modified
# in place, so they are shared; arrays of numbers are copied as containers, which keeps the
# unassigned entries of fresh buffers (`Vector{BigFloat}(undef, n)`) unassigned.
copy_working(x::Number) = x
copy_working(x::AbstractArray{<:Number}) = copy(x)
copy_working(x::AbstractArray) = isbitstype(eltype(x)) ? copy(x) : map(copy_working, x)
# DataStructures' heap holds only numbers
copy_working(x::MutableBinaryMinHeap) = deepcopy(x)
# Values of bits types are copied by value. Any other type must declare its `shared_fields`,
# so that a field nobody classified is copied, or makes `isolated_copy` throw, but is never
# shared by mistake.
copy_working(x) = isbits(x) ? x : copy_working_state(x)

fresh_aggregator(::Nothing) = nothing
fresh_aggregator(agg::AbstractSSAJumpAggregator) = copy_working_state(agg)

# The jump callbacks of a copy, rebuilt around its new aggregator. `vjumps` are the variable
# rate jumps that the `VR_Direct` and `VR_DirectFW` caches handle.
function isolated_jump_callback(cbs::CallbackSet, agg, vjumps)
    CallbackSet(
        map(cb -> isolated_jump_callback(cb, agg, vjumps), cbs.continuous_callbacks)...,
        map(cb -> isolated_jump_callback(cb, agg, vjumps), cbs.discrete_callbacks)...)
end
function isolated_jump_callback(cb::Union{DiscreteCallback, ContinuousCallback}, agg, vjumps)
    isolated_jump_callback(cb.condition, cb, agg, vjumps)
end
isolated_jump_callback(cb, agg, vjumps) = cb
function isolated_jump_callback(::AbstractSSAJumpAggregator, ::DiscreteCallback, agg, vjumps)
    DiscreteCallback(agg)
end
function isolated_jump_callback(cache::VR_DirectEventCache, ::ContinuousCallback, agg,
        vjumps)
    build_variable_integcallback(copy_working_state(cache), vjumps)
end
# `VR_FRM`'s callbacks keep their state in the integrator's `u.jump_u`, so they are shared
isolated_jump_callback(condition, cb, agg, vjumps) = cb

"""
    isolated_copy(jprob::JumpProblem)

Return a `JumpProblem` with its own jump state, sharing everything else with `jprob`.

The copy shares no state that JumpProcesses' solvers modify, so it can be solved
concurrently with `jprob`, or used for an integrator that is alive at the same time as one of
`jprob`'s. Since it does not copy the wrapped problem or the jumps, it is much cheaper than
`deepcopy(jprob)`, especially for problems built with ModelingToolkit or Catalyst.

The copy has a new jump aggregator, with copies of the aggregator's working data (such as the
current rates and its priority queue or table), and new jump callbacks built around it,
including a new cache for `VR_Direct` and `VR_DirectFW`. It shares everything else with
`jprob`:

  - the wrapped problem `jprob.prob`, including its function, `u0`, `p` and `tspan`;
  - the jumps, with their rate and affect functions;
  - the dependency graphs;
  - the callbacks and `tstops` passed to the `JumpProblem` constructor.

The copy starts from `jprob`'s current working data, which solvers re-initialize at every
`init`, so it doesn't matter whether `jprob` was solved before. The one exception is
`SortingDirect`'s learned search order, which the copy keeps.

A copy may be taken between `step!` calls of an integrator of `jprob`, in the same task. Do
not take one while another thread is solving `jprob`.

## What the copy does not isolate

  - `p` and `u0`. Solvers may alias them, so a callback that modifies `integrator.p` can also
    modify `jprob.prob.p`. Use `remake(isolated_copy(jprob); p = new_p)` to give the copy its
    own parameters. With `alias_u0 = true`, an integrator's state is `jprob.prob.u0`, so a copy
    initialized after it has run starts from that integrator's current state.
  - State captured by rate, affect or callback functions. Pass callbacks with such state to
    `init` or `solve` instead of storing them in the problem.
  - A noise process passed to an `SDEProblem` with `noise`. StochasticDiffEq copies each
    solve's noise back into `prob.noise`, so copies that share it must not be solved
    concurrently. Give a copy its own with
    `remake(isolated_copy(jprob); prob = remake(jprob.prob; noise = new_noise))`.

`deepcopy(jprob)` also copies `p`, `u0`, the stored callbacks and the wrapped problem, at a
much higher cost; see [`JumpProblem`](@ref) for its limitations.

## Example

```julia
# one copy per task, each with its own parameters
tasks = map([1.0, 2.0, 3.0]) do birth_rate
    Threads.@spawn solve(remake(isolated_copy(jprob); p = [birth_rate, 0.1]), SSAStepper())
end
sols = fetch.(tasks)
```
"""
function isolated_copy(jprob::JumpProblem)
    agg = fresh_aggregator(jprob.discrete_jump_aggregation)
    jump_callback = isolated_jump_callback(jprob.jump_callback, agg, jprob.variable_jumps)
    remaker_of(jprob)(jprob.prob, jprob.aggregator, agg, jump_callback,
        jprob.constant_jumps, jprob.variable_jumps, jprob.regular_jump,
        jprob.massaction_jump, jprob.kwargs)
end
