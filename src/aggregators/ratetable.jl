# dynamic table data structure to store and update rates

# one group of rates
mutable struct RateGroup{S,T,U}
    gid::S
    gmin::T
    gmax::T
    numrates::Int
    jidxs::AbstractVector{U}
    RateGroup{S,T,U}(gid::S, gmin::T, gmax::T) = new{S,T,U}()
end



# table of groups
mutable struct RateTable{F,G,S}
    minrate::F
    maxrate::F
    groups::Vector{G}
    gsums::Vector{F}
    jtogroup::Vector{Tuple{S,S}}
end

function rebuild!(rt::RateTable, newminrate, newmaxrate)

    nothing
end

function insert!(rt::RateTable, jumpid, rate)

    nothing
end

function update!(rt::RateTable, jumpid, oldrate, newrate)

    nothing
end

function samplerx(rt::RateTable, sampleval)

end

function samplerx(rt::RateGroup)

end