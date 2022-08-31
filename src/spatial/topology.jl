"""
A file with structs and functions for setting up and using the topology of the spatial system (e.g. a graph or a Cartesian grid)
"""

################### Graph ########################
num_sites(graph::AbstractGraph) = Graphs.nv(graph)
# neighbors(graph::AbstractGraph, site) = Graphs.neighbors(graph, site)
# outdegree(graph::AbstractGraph, site) = Graphs.outdegree(graph, site)
rand_nbr(rng, graph::AbstractGraph, site) = rand(rng, neighbors(graph, site))
nth_nbr(graph::AbstractGraph, site, n) = @inbounds neighbors(graph, site)[n]

################### CartesianGrid ########################
const offsets_1D = [CartesianIndex(-1), CartesianIndex(1)]
const offsets_2D = [
    CartesianIndex(0, -1),
    CartesianIndex(-1, 0),
    CartesianIndex(1, 0),
    CartesianIndex(0, 1),
]
const offsets_3D = [
    CartesianIndex(0, 0, -1),
    CartesianIndex(0, -1, 0),
    CartesianIndex(-1, 0, 0),
    CartesianIndex(1, 0, 0),
    CartesianIndex(0, 1, 0),
    CartesianIndex(0, 0, 1),
]

"""
    potential_offsets(dimension::Int)

NOTE: dimension is assumed to be 1, 2, or 3
"""
function potential_offsets(dimension::Int)
    if dimension == 1
        return offsets_1D
    elseif dimension == 2
        return offsets_2D
    else # else dimension == 3
        return offsets_3D
    end
end

dimension(grid) = length(grid.dims)
num_sites(grid) = prod(grid.dims)
outdegree(grid, site) = grid.nums_neighbors[site]

"""
    nth_potential_nbr(grid, site, n)

return nth Cartesian neighbor ignoring boundaries
"""
nth_potential_nbr(grid, site, n) = grid.LI[grid.offsets[n] + grid.CI[site]]

"""
    nth_nbr(grid, site, n)

return the nth neighbor of site in grid, in ascending order
"""
function nth_nbr(grid, site, n)
    CI = grid.CI
    offsets = grid.offsets
    @inbounds I = CI[site]
    @inbounds for off in offsets
        nb = I + off
        if nb in CI
            n -= 1
            @inbounds n == 0 && return grid.LI[nb]
        end
    end
end

"""
    neighbors(grid, site)

return an iterator over neighbors of site in ascending order. Do not use in hot loops
"""
function neighbors(grid, site)
    CI = grid.CI
    LI = grid.LI
    I = CI[site]
    Iterators.map(off -> LI[off + I], Iterators.filter(off -> off + I in CI, grid.offsets))
end

"""
given a vector of hopping constants of length num_neighbors(grid, site), form the vector of size 2*(dimension of grid) with zeros at indices where the neighbor is out of bounds. Store it in to_pad
"""
function pad_hop_vec!(to_pad::AbstractVector{F}, grid, site,
                      hop_vec::Vector{F}) where {F <: Number}
    CI = grid.CI
    I = CI[site]
    nbr_counter = 1
    for (i, off) in enumerate(grid.offsets)
        if I + off in CI
            to_pad[i] = hop_vec[nbr_counter]
            nbr_counter += 1
        else
            to_pad[i] = zero(F)
        end
    end
    to_pad
end

CartesianGrid(dims) = CartesianGridRej(dims) # use CartesianGridRej by default

# neighbor sampling is rejection-based
struct CartesianGridRej{N, T}
    "dimensions (side lengths) of the grid"
    dims::NTuple{N, Int}

    "number of neighbor for each site"
    nums_neighbors::Vector{Int8}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}

    "offsets, e.g. [-1, 1] for 1D"
    offsets::Vector{CartesianIndex{N}}
end

"""
    CartesianGridRej(dims::Tuple)

initialze CartesianGridRej
"""
function CartesianGridRej(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = Int8[count(x -> x + CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGridRej(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGridRej(dims) = CartesianGridRej(Tuple(dims))
function CartesianGridRej(dimension, linear_size::Int)
    CartesianGridRej([linear_size for i in 1:dimension])
end
function rand_nbr(rng, grid::CartesianGridRej, site::Int)
    CI = grid.CI
    offsets = grid.offsets
    @inbounds I = CI[site]
    while true
        @inbounds nb = rand(rng, offsets) + I
        @inbounds nb in CI && return grid.LI[nb]
    end
end

# neighbor sampling is iterator-based
struct CartesianGridIter{N, T}
    dims::NTuple{N, Int}
    nums_neighbors::Vector{Int8}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
end
function CartesianGridIter(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = Int8[count(x -> x + CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGridIter(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGridIter(dims) = CartesianGridIter(Tuple(dims))
function CartesianGridIter(dimension, linear_size::Int)
    CartesianGridIter([linear_size for i in 1:dimension])
end
function rand_nbr(rng, grid::CartesianGridIter, site::Int)
    nth_nbr(grid, site, rand(rng, 1:outdegree(grid, site)))
end

function Base.show(io::IO, ::MIME"text/plain",
                   grid::CartesianGridRej)
    println(io, "A Cartesian grid with dimensions $(grid.dims)")
end
