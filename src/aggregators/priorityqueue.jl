# This file contains code that was formerly a part of Julia. License is MIT: http://julialang.org/license

# slightly optimized version of DataStructure.jl priority queue
# swapped index field from a dict to an array
# removed checking for presence of keys when inserting

# Binary heap indexing
heapleft(i::Int)   = i << 1         #2i 
heapright(i::Int)  = (i << 1) + 1   #2i + 1
heapparent(i::Int) = i >> 1         #div(i, 2)

function not_iterator_of_pairs(kv)
    return any(x->isempty(methodswith(typeof(kv), x, true)),
               [start, next, done]) ||
           any(x->!isa(x, Union{Tuple,Pair}), kv)
end

mutable struct ArrayPQ{K,V,O<:Ordering} <: AbstractDict{K,V}
    # Binary heap of (element, priority) pairs.
    xs::Vector{Pair{K,V}}
    o::O

    # Map elements to their index in xs
    index::Vector{Int}

    function ArrayPQ{K,V,O}(o::O, itr) where {K,V,O<:Ordering}
        xs    = Vector{Pair{K,V}}(undef, length(itr))
        index = Vector{Int}(length(itr))
        @inbounds for (i,p) in enumerate(itr)
            xs[i]    = p
            index[i] = i
        end
        pq = new{K,V,O}(xs, o, index)

        # heapify
        for i in heapparent(length(pq.xs)):-1:1
            percolate_down!(pq, i)
        end

        pq
    end
end
ArrayPQ(kv::Vector{T}, o::Ordering=Forward) where {K,V,T <: Pair{K,V}} = ArrayPQ{K,V,typeof(o)}(o, kv)   
ArrayPQ(kv, o::Ordering=Forward) = ArrayPQ(o, kv)

length(pq::ArrayPQ)  = length(pq.xs)
isempty(pq::ArrayPQ) = isempty(pq.xs)
peek(pq::ArrayPQ)    = (@inbounds return pq.xs[1])

function percolate_down!(pq::ArrayPQ, i::Integer)
    @inbounds x = pq.xs[i]
    @inbounds while (l = heapleft(i)) <= length(pq)
        r = heapright(i)
        j = r > length(pq) || lt(pq.o, pq.xs[l].second, pq.xs[r].second) ? l : r
        if lt(pq.o, pq.xs[j].second, x.second)
            pq.index[pq.xs[j].first] = i
            pq.xs[i] = pq.xs[j]
            i = j
        else
            break
        end
    end
    @inbounds pq.index[x.first] = i
    @inbounds pq.xs[i] = x
    nothing
end

function percolate_up!(pq::ArrayPQ, i::Integer)
    @inbounds x = pq.xs[i]
    @inbounds while i > 1
        j = heapparent(i)
        if lt(pq.o, x.second, pq.xs[j].second)
            pq.index[pq.xs[j].first] = i
            pq.xs[i] = pq.xs[j]
            i = j
        else
            break
        end
    end
    @inbounds pq.index[x.first] = i
    @inbounds pq.xs[i] = x
    nothing    
end

function getindex(pq::ArrayPQ{K,V}, key) where {K,V}
    @inbounds return pq.xs[pq.index[key]].second
end

# Change the priority of an existing element
# does not check if the element is present
function setindex!(pq::ArrayPQ{K, V}, value, key) where {K,V}
    @inbounds i = pq.index[key]
    @inbounds oldvalue = pq.xs[i].second
    @inbounds pq.xs[i] = Pair{K,V}(key, value)
    if lt(pq.o, oldvalue, value)
        percolate_down!(pq, i)
    else
        percolate_up!(pq, i)
    end
    value
end

# Unordered iteration through key value pairs in a ArrayPQ
start(pq::ArrayPQ) = start(pq.index)
done(pq::ArrayPQ, i) = done(pq.index, i)
function next(pq::ArrayPQ{K,V}, i) where {K,V}
    (k, idx), i = next(pq.index, i)
    return (pq.xs[idx], i)
end
