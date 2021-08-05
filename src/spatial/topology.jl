############ spatial system interface ##################

# num_sites(spatial_system) = total number of sites
# neighbors(spatial_system, site) = an iterator over the neighbors of site
# num_neighbors(spatial_system, site) = number of neighbors of site

################### LightGraph ########################
num_sites(graph::AbstractGraph) = LightGraphs.nv(graph)
# neighbors(graph::AbstractGraph, site) = LightGraphs.neighbors(graph, site)
num_neighbors(graph::AbstractGraph, site) = LightGraphs.outdegree(graph, site)
rand_nbr(graph::AbstractGraph, site) = rand(neighbors(graph, site))
nth_nbr(graph::AbstractGraph, site, n) = neighbors(graph, site)[n]

################### CartesianGrid ########################
const offsets_1D = [CartesianIndex(-1),CartesianIndex(1)]
const offsets_2D = [CartesianIndex(0,-1),CartesianIndex(-1,0),CartesianIndex(1,0),CartesianIndex(0,1)]
const offsets_3D = [CartesianIndex(0,0,-1), CartesianIndex(0,-1,0),CartesianIndex(-1,0,0),CartesianIndex(1,0,0),CartesianIndex(0,1,0),CartesianIndex(0,0,1)]
"""
dimension is assumed to be 1, 2, or 3
"""
function potential_offsets(dimension::Int)
    if dimension==1
        return offsets_1D
    elseif dimension==2
        return offsets_2D
    else # else dimension == 3
        return offsets_3D
    end
end

num_sites(grid) = prod(grid.dims)
num_neighbors(grid, site) = grid.nums_neighbors[site]
function nth_nbr(grid, site, n)
    CI = grid.CI; offsets = grid.offsets
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
return neighbors of site in increasing order
"""
function neighbors(grid, site)
    CI = grid.CI
    LI = grid.LI
    I = CI[site]
    Iterators.map(off -> LI[off+I], Iterators.filter(off -> off+I in CI, grid.offsets))
end

# possible rand_nbr functions:
# 1. Rejection-based: pick a neighbor, check it's valid; if not, repeat
# 2. Iterator-based: draw a random number from 1 to num_neighbors and iterate to that neighbor
# 3. Array-based: using a pre-allocated array in grid

# rejection-based
struct CartesianGridRej{N,T}
    dims::NTuple{N, Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
end
function CartesianGridRej(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = [count(x -> x+CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGridRej(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGridRej(dims) = CartesianGridRej(Tuple(dims))
CartesianGridRej(dimension, linear_size::Int) = CartesianGridRej([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGridRej, site::Int)
    CI = grid.CI; offsets = grid.offsets
    @inbounds I = CI[site]
    while true
        @inbounds nb = rand(offsets) + I
        @inbounds nb in CI && return grid.LI[nb]
    end
end

# iterator-based
struct CartesianGridIter{N,T}
    dims::NTuple{N, Int} #side lengths of the grid
    nums_neighbors::Vector{Int}
    CI::CartesianIndices{N, T}
    LI::LinearIndices{N, T}
    offsets::Vector{CartesianIndex{N}}
end
function CartesianGridIter(dims::Tuple)
    dim = length(dims)
    CI = CartesianIndices(dims)
    LI = LinearIndices(dims)
    offsets = potential_offsets(dim)
    nums_neighbors = [count(x -> x+CI[site] in CI, offsets) for site in 1:prod(dims)]
    CartesianGridIter(dims, nums_neighbors, CI, LI, offsets)
end
CartesianGridIter(dims) = CartesianGridIter(Tuple(dims))
CartesianGridIter(dimension, linear_size::Int) = CartesianGridIter([linear_size for i in 1:dimension])
function rand_nbr(grid::CartesianGridIter, site::Int)
    nth_nbr(grid, site, rand(1:num_neighbors(grid,site)))
end

function Base.show(io::IO, grid::Union{CartesianGridRej, CartesianGridIter})
    println(io, "A Cartesian grid with dimensions $(grid.dims)")
end