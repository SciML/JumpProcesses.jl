"""
    JumpDependencies(; vartojumps_map = nothing, jumptovars_map = nothing,
        dep_graph = nothing, vrjtovars_map = nothing, vrj_dep_graph = nothing,
        alias = false)

How the jumps handled by an aggregator depend on each other and on the system's variables,
passed to a [`JumpProblem`](@ref) with its `dependencies` keyword. Aggregators that update only
the rates a jump affects use these maps: `SortingDirect`, `RDirect`, `NRM`, `CCNRM`,
`DirectCR` and `Coevolve` use `dep_graph`, `RSSA` and `RSSACR` use `vartojumps_map` and
`jumptovars_map`, and the spatial aggregators use all three. Other aggregators ignore them.

Each map is a collection whose `i`th entry is an `AbstractVector` of integer indices, for
example a `Vector{Vector{Int}}`. A map not given is `nothing`.

  - `dep_graph`: entry `i` lists the jumps whose rates must be recalculated when jump `i`
    occurs.
  - `vartojumps_map`: entry `i` lists the jumps whose rates depend on the variable `u[i]`.
  - `jumptovars_map`: entry `i` lists the variables whose values jump `i` changes.
  - `vrjtovars_map`: entry `k` lists the variables whose values the `k`th `VariableRateJump`
    changes.
  - `vrj_dep_graph`: entry `k` lists the jumps whose rates must be recalculated when the
    `k`th `VariableRateJump` occurs.

Jumps are numbered in the order aggregators use: `MassActionJump`s first, then
`ConstantRateJump`s, then bounded `VariableRateJump`s when the aggregator handles them (as
`Coevolve` does), each in the order given to the `JumpProblem`. General `VariableRateJump`s
are not numbered among them. For the spatial aggregators the maps describe one site: the
variables are the species, and the jumps the reactions of the spatial `MassActionJump`. In
`vrjtovars_map` and `vrj_dep_graph`, `VariableRateJump`s are numbered in the order given to
the `JumpProblem`; the entries of bounded ones that the aggregator handles are ignored.

## `VariableRateJump`s

Without `vrjtovars_map` and `vrj_dep_graph`, the aggregator is not told when a general
`VariableRateJump` occurs, so the rate of a jump it handles must not depend on a variable
that a general `VariableRateJump` changes. Giving either map lifts that rule: after each
`VariableRateJump` occurs, the aggregator recomputes the rates that depend on the change.
What each aggregator needs:

  - `Direct`, `DirectFW`, `FRM` and `FRMFW` recompute every rate after a
    `VariableRateJump` occurs, except those that `vrj_dep_graph` shows affect no rate.
  - `SortingDirect`, `RDirect`, `NRM`, `CCNRM`, `DirectCR` and `Coevolve` need
    `vrj_dep_graph`, or `vrjtovars_map` and `vartojumps_map` (generated when every jump is a
    `MassActionJump`).
  - `RSSA` and `RSSACR` need `vrjtovars_map`.

Listing every jump in each entry of `vrj_dep_graph` gives exact updates without working out
the dependencies, at a cost in proportion to the number of jumps each time a
`VariableRateJump` occurs.

## Checks and normal form

A `JumpProblem` checks the lengths and indices of the maps against the problem (except for
spatial problems), and puts each map its aggregator uses into one form: every entry sorted,
without duplicates and with `Int` indices, and each entry of `dep_graph` listing its own
jump. A map already in that form is used as is, without a copy. When every jump is a
`MassActionJump`, a map the aggregator needs but wasn't given is generated from them.

## Keyword arguments

  - `alias = false`: with the default the maps are copied, and the given maps are never
    changed. With `alias = true` the `JumpDependencies` holds the given maps, and
    JumpProcesses may modify them; don't use or change them afterwards.

A `JumpProblem` takes ownership of the `JumpDependencies` passed to it, and may put its maps
into the form above in place. Afterwards it belongs to that problem and its `remake`s, and is
not changed again. To keep it unchanged, for example to build several problems from it, pass
`alias_dependencies = false` to the `JumpProblem`. `copy(deps)` copies the maps.

The `JumpProblem` keywords `vartojumps_map`, `jumptovars_map` and `dep_graph` are deprecated.
Maps given that way are not checked, and JumpProcesses may modify them.
"""
struct JumpDependencies{V, J, D, VV, VD}
    vartojumps_map::V
    jumptovars_map::J
    dep_graph::D
    vrjtovars_map::VV
    vrj_dep_graph::VD
end

function JumpDependencies(; vartojumps_map = nothing, jumptovars_map = nothing,
        dep_graph = nothing, vrjtovars_map = nothing, vrj_dep_graph = nothing,
        alias = false)
    if !alias
        vartojumps_map = copy_map(vartojumps_map)
        jumptovars_map = copy_map(jumptovars_map)
        dep_graph = copy_map(dep_graph)
        vrjtovars_map = copy_map(vrjtovars_map)
        vrj_dep_graph = copy_map(vrj_dep_graph)
    end
    JumpDependencies(vartojumps_map, jumptovars_map, dep_graph, vrjtovars_map,
        vrj_dep_graph)
end

function Base.copy(deps::JumpDependencies)
    JumpDependencies(copy_map(deps.vartojumps_map), copy_map(deps.jumptovars_map),
        copy_map(deps.dep_graph), copy_map(deps.vrjtovars_map),
        copy_map(deps.vrj_dep_graph))
end

copy_map(::Nothing) = nothing
copy_map(map) = Base.map(copy, map)

# The indices listed in entry `i` of a map. The maps are read through this, so that their
# storage can change later.
Base.@propagate_inbounds dependents(map, i) = map[i]

const DEPENDENCY_KEYWORDS_DEPWARN = "Passing `dep_graph`, `vartojumps_map` or \
    `jumptovars_map` as `JumpProblem` keywords is deprecated, and these keywords will be \
    removed in JumpProcesses 11. Pass the maps in a `JumpDependencies` instead, as in \
    `JumpProblem(prob, aggregator, jumps...; dependencies = JumpDependencies(; dep_graph))`."

# Merges the maps given in `dependencies` with those given as `JumpProblem` keywords, which
# are used as given (aliased), as before `JumpDependencies` existed.
function merge_dependencies(dependencies, vartojumps_map, jumptovars_map, dep_graph)
    deps = dependencies === nothing ? JumpDependencies(; alias = true) : dependencies
    JumpDependencies(merge_map(deps.vartojumps_map, vartojumps_map, :vartojumps_map),
        merge_map(deps.jumptovars_map, jumptovars_map, :jumptovars_map),
        merge_map(deps.dep_graph, dep_graph, :dep_graph), deps.vrjtovars_map,
        deps.vrj_dep_graph)
end

merge_map(::Nothing, ::Nothing, name) = nothing
merge_map(map, ::Nothing, name) = map
merge_map(::Nothing, map, name) = map
function merge_map(map1, map2, name)
    throw(ArgumentError("`$name` was given both in `dependencies` and as a `JumpProblem` \
        keyword. Give it only one way."))
end

# The keywords through which the aggregators receive the maps
function dependency_kwargs(deps::JumpDependencies)
    (; deps.vartojumps_map, deps.jumptovars_map, deps.dep_graph)
end

# Checks the maps of a `JumpDependencies` built by the user against a problem with
# `num_vars` variables, `num_jumps` jumps handled by its aggregator and `num_vrjs`
# `VariableRateJump`s.
function check_dependencies(deps::JumpDependencies, num_vars, num_jumps, num_vrjs)
    check_map(deps.dep_graph, :dep_graph, "jump", num_jumps, "jump", num_jumps)
    check_map(deps.vartojumps_map, :vartojumps_map, "variable", num_vars, "jump",
        num_jumps)
    check_map(deps.jumptovars_map, :jumptovars_map, "jump", num_jumps, "variable",
        num_vars)
    check_map(deps.vrjtovars_map, :vrjtovars_map, "VariableRateJump", num_vrjs,
        "variable", num_vars)
    check_map(deps.vrj_dep_graph, :vrj_dep_graph, "VariableRateJump", num_vrjs, "jump",
        num_jumps)
    nothing
end

check_map(::Nothing, name, entry, num_entries, index, num_indices) = nothing
function check_map(map, name, entry, num_entries, index, num_indices)
    if length(map) != num_entries
        throw(ArgumentError("`$name` has $(length(map)) entries, but needs one per $entry: \
            the problem has $num_entries $(entry)s."))
    end
    for i in 1:num_entries
        for j in dependents(map, i)
            if !(j isa Integer && 1 <= j <= num_indices)
                throw(ArgumentError("Entry $i of `$name` lists $j, which is not a $index \
                    index: the problem has $num_indices $(index)s."))
            end
        end
    end
    nothing
end

# The number of variables a problem's maps index
num_variables(u0) = length(u0)
num_variables(u0::ExtendedJumpArray) = length(u0.u)

############################## Normalization #################################

# the maps each aggregator uses
uses_dep_graph(aggregator) = needs_depgraph(aggregator) || is_spatial(aggregator)
uses_species_maps(aggregator) = needs_vartojumps_map(aggregator) || is_spatial(aggregator)

# Returns the maps the aggregator uses, each in normal form: every entry sorted and
# without duplicates, with `Int` indices, and each `dep_graph` entry listing its own jump.
# A map already in that form is used as is. Otherwise it is fixed in place if `modifiable`
# allows it for that map and its entries are `Vector{Int}`s, and rebuilt if not. A missing
# map is generated from the mass action jumps `maj` when every jump is one (always for the
# spatial aggregators), and is an error otherwise. The problem has `num_vars` variables
# and `num_jumps` jumps handled by the aggregator, and `other_jumps` names its other kinds
# of jumps (for the errors). Maps the aggregator doesn't use are left out (`nothing`), so
# that every map of the result is in normal form.
function normalize_dependencies(aggregator, deps::JumpDependencies, modifiable, maj,
        num_vars, num_jumps, other_jumps = "`ConstantRateJump`s")
    vartojumps_map = jumptovars_map = dep_graph = nothing
    num_majumps = get_num_majumps(maj)
    can_generate = is_spatial(aggregator) || (num_majumps > 0 && num_jumps == num_majumps)
    if uses_dep_graph(aggregator)
        if deps.dep_graph === nothing
            can_generate || error(missing_map_message(aggregator, "a dependency graph",
                "for $other_jumps", "`dep_graph`"))
            dep_graph = normalize_map(generated_dep_graph(num_vars, maj), true, true)
        else
            dep_graph = normalize_map(deps.dep_graph, true, modifiable.dep_graph)
        end
    end
    if uses_species_maps(aggregator)
        if deps.vartojumps_map === nothing
            can_generate || error(missing_map_message(aggregator,
                "a map from variables to dependent jumps", "for $other_jumps",
                "`vartojumps_map`"))
            vartojumps_map = normalize_map(generated_vartojumps_map(num_vars, maj), false,
                true)
        else
            vartojumps_map = normalize_map(deps.vartojumps_map, false,
                modifiable.vartojumps_map)
        end
        if deps.jumptovars_map === nothing
            can_generate || error(missing_map_message(aggregator,
                "a map from jumps to dependent variables", "for $other_jumps",
                "`jumptovars_map`"))
            jumptovars_map = normalize_map(generated_jumptovars_map(maj), false, true)
        else
            jumptovars_map = normalize_map(deps.jumptovars_map, false,
                modifiable.jumptovars_map)
        end
    end
    JumpDependencies(vartojumps_map, jumptovars_map, dep_graph, nothing, nothing)
end

generated_dep_graph(num_vars, ::Nothing) = Vector{Int}[]
generated_dep_graph(num_vars, maj) = make_dependency_graph(num_vars, maj)
generated_vartojumps_map(num_vars, ::Nothing) = [Int[] for _ in 1:num_vars]
generated_vartojumps_map(num_vars, maj) = var_to_jumps_map(num_vars, maj)
generated_jumptovars_map(::Nothing) = Vector{Int}[]
generated_jumptovars_map(maj) = jump_to_vars_map(maj)

# The error message for a map the aggregator needs that wasn't given and can't be
# generated: `map` describes it, `purpose` says what it is needed for, and `fields` names
# the fields of a `JumpDependencies` that give it.
function missing_map_message(aggregator, map, purpose, fields)
    "The $(nameof(typeof(aggregator))) algorithm needs $map $purpose: give $fields in the \
    `JumpDependencies`."
end

# the map in normal form; entry `i` of a map with `addself` must list `i`
function normalize_map(map, addself, modifiable)
    isnormal_map(map, addself) && return map
    if modifiable && all(entry -> entry isa Vector{Int}, map)
        for i in 1:length(map)
            normalize_entry!(dependents(map, i), addself ? i : 0)
        end
        return map
    end
    # `Vector{Int}` rows: resizable for any input row (`collect` keeps static vectors static)
    [normalize_entry!(Vector{Int}(dependents(map, i)), addself ? i : 0)
     for i in 1:length(map)]
end

# Whether every entry has `Int` indices (the heap-based aggregators use them as handles), is
# sorted and without duplicates, and lists its own index if `addself`.
function isnormal_map(map, addself)
    for i in 1:length(map)
        entry = dependents(map, i)
        (eltype(entry) === Int && isnormal_entry(entry, addself ? i : 0)) || return false
    end
    true
end

# whether the entry is sorted, without duplicates, and lists `self` (when positive)
function isnormal_entry(entry, self)
    hasself = self <= 0
    isfirst = true
    prev = 0
    for j in entry
        (isfirst || j > prev) || return false
        isfirst = false
        prev = j
        hasself |= (j == self)
    end
    hasself
end

function normalize_entry!(entry, self)
    (self > 0 && !(self in entry)) && push!(entry, self)
    unique!(sort!(entry))
end

# The lists the VR aggregator's callbacks hand to the SSA aggregator after a
# `VariableRateJump` occurs, indexed by its position `k` among the jumps the VR aggregator
# handles (`cvrj_idxs` holds their positions among all `VariableRateJump`s): the jumps whose
# rates to recompute, or for RSSA and RSSACR the changed variables that some jump reads.
# Returns `nothing` when no `VariableRateJump` map was given (no updates, as before), and
# `VRJUpdates(nothing)` when the aggregator recomputes every rate and the dependents aren't
# known. `deps` holds the maps as given, `agg_deps` the aggregator's normalized maps.
function vrj_updates(aggregator, deps::JumpDependencies, agg_deps::JumpDependencies, maj,
        num_vars, num_jumps, cvrj_idxs)
    (deps.vrjtovars_map === nothing && deps.vrj_dep_graph === nothing) && return nothing
    isempty(cvrj_idxs) && return nothing
    if uses_species_maps(aggregator)
        deps.vrjtovars_map === nothing &&
            error(missing_map_message(aggregator,
                "the variables each `VariableRateJump` changes",
                "to update its jumps after one occurs", "`vrjtovars_map`"))
        v2j = agg_deps.vartojumps_map
        lists = [filter!(v -> !isempty(dependents(v2j, v)),
                     normal_vrj_entry(dependents(deps.vrjtovars_map, k)))
                 for k in cvrj_idxs]
        return VRJUpdates(lists)
    end
    num_majumps = get_num_majumps(maj)
    can_generate = num_majumps > 0 && num_jumps == num_majumps
    lists = vrj_jump_lists(deps, maj, num_vars, can_generate, cvrj_idxs)
    if lists === nothing && needs_depgraph(aggregator)
        error(missing_map_message(aggregator,
            "the jumps that depend on each `VariableRateJump`",
            "to update them after it occurs",
            "`vrj_dep_graph`, or `vrjtovars_map` and `vartojumps_map`,"))
    end
    VRJUpdates(lists)
end

# Each handled `VariableRateJump`'s dependent jumps, from `vrj_dep_graph` or derived from
# `vrjtovars_map` and `vartojumps_map`; `nothing` when neither way is possible.
function vrj_jump_lists(deps, maj, num_vars, can_generate, cvrj_idxs)
    if deps.vrj_dep_graph !== nothing
        return [normal_vrj_entry(dependents(deps.vrj_dep_graph, k)) for k in cvrj_idxs]
    end
    v2j = deps.vartojumps_map
    if v2j === nothing
        can_generate || return nothing
        v2j = generated_vartojumps_map(num_vars, maj)
    end
    map(cvrj_idxs) do k
        list = Int[]
        for v in dependents(deps.vrjtovars_map, k)
            append!(list, dependents(v2j, v))
        end
        unique!(sort!(list))
    end
end

normal_vrj_entry(entry) = unique!(sort!(Vector{Int}(entry)))
