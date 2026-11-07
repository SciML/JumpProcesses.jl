"""
    JumpDependencies(; vartojumps_map = nothing, jumptovars_map = nothing,
        dep_graph = nothing, alias = false)

How the jumps handled by an aggregator depend on each other and on the system's variables,
passed to a [`JumpProblem`](@ref) with its `dependencies` keyword. Aggregators that update only
the rates a jump affects use these maps: `SortingDirect`, `RDirect`, `NRM`, `CCNRM`,
`DirectCR`, `Coevolve` and the spatial aggregators use `dep_graph`, and `RSSA` and `RSSACR` use
`vartojumps_map` and `jumptovars_map`. Other aggregators ignore them.

Each map is an adjacency list: a collection whose `i`th entry is a collection of integer
indices, for example a `Vector{Vector{Int}}`. A map not given is `nothing`.

  - `dep_graph`: entry `i` lists the jumps whose rates must be recalculated when jump `i`
    occurs.
  - `vartojumps_map`: entry `i` lists the jumps whose rates depend on the variable `u[i]`.
  - `jumptovars_map`: entry `i` lists the variables whose values jump `i` changes.

Jumps are numbered in the order aggregators use: `MassActionJump`s first, then
`ConstantRateJump`s, then bounded `VariableRateJump`s when the aggregator handles them (as
`Coevolve` does), each in the order given to the `JumpProblem`. General `VariableRateJump`s
are not numbered. The `JumpProblem` checks the maps' lengths and indices against the problem.

## Keyword arguments

  - `alias = false`: with the default the maps are copied, and the given maps are never
    changed. With `alias = true`, JumpProcesses may use the given maps directly and modify
    them, for example to add each jump to its own entry of `dep_graph`; don't use or change
    them afterwards.

The `JumpProblem` keywords `vartojumps_map`, `jumptovars_map` and `dep_graph` give maps as
`alias = true` does. A map can't be given both there and in `dependencies`.
"""
struct JumpDependencies{V, J, D}
    vartojumps_map::V
    jumptovars_map::J
    dep_graph::D
end

function JumpDependencies(; vartojumps_map = nothing, jumptovars_map = nothing,
        dep_graph = nothing, alias = false)
    if !alias
        vartojumps_map = copy_map(vartojumps_map)
        jumptovars_map = copy_map(jumptovars_map)
        dep_graph = copy_map(dep_graph)
    end
    JumpDependencies(vartojumps_map, jumptovars_map, dep_graph)
end

copy_map(::Nothing) = nothing
copy_map(map) = Base.map(copy, map)

# The indices listed in entry `i` of a map. New code reads the maps through this, so that
# their storage can change later.
dependents(map, i) = map[i]

# Merges the maps given in `dependencies` with those given as `JumpProblem` keywords, which
# are used as given (aliased), as before `JumpDependencies` existed.
function merge_dependencies(dependencies, vartojumps_map, jumptovars_map, dep_graph)
    deps = dependencies === nothing ? JumpDependencies(; alias = true) : dependencies
    JumpDependencies(merge_map(deps.vartojumps_map, vartojumps_map, :vartojumps_map),
        merge_map(deps.jumptovars_map, jumptovars_map, :jumptovars_map),
        merge_map(deps.dep_graph, dep_graph, :dep_graph))
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
# `num_vars` variables and `num_jumps` jumps handled by its aggregator.
function check_dependencies(deps::JumpDependencies, num_vars, num_jumps)
    check_map(deps.dep_graph, :dep_graph, "jump", num_jumps, "jump", num_jumps)
    check_map(deps.vartojumps_map, :vartojumps_map, "variable", num_vars, "jump",
        num_jumps)
    check_map(deps.jumptovars_map, :jumptovars_map, "jump", num_jumps, "variable",
        num_vars)
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
