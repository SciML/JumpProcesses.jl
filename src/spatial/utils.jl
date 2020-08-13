"""
A file with utility functions for spatial stuff
"""

"return coordinates from node index and number of sites per box edge"
node_to_coordinates(j,m) = ( (j-1)%m+1,(div(j-1,m))%m+1,(div(j-1,m^2)+1) )

"return node index from coordinates and number of sites per box edge"
coordinates_to_node(x,y,z,m) = x + (y-1)*m + (z-1)*m^2

"return node index from coordinates and number of sites per box edge"
coordinates_to_node(x,y,m) = x + (y-1)*m

"return node index from coordinates and number of sites per box edge"
coordinates_to_node(x,m) = x

function connectivity_list_from_box(box_width :: Integer, dimension :: Integer)
    @assert 1 <= dimension <= 3
    vol_num = box_width^dimension
    connectivity_matrix = Array{Array{Int64,1},1}(undef, 0)
    for j in 1:vol_num
        x,y,z = node_to_coordinates(j, box_width)
        if dimension == 1
            potential_neighbors = [(x-1,y,z), (x+1,y,z)]
        elseif dimension == 2
            potential_neighbors = [(x-1,y,z), (x+1,y,z), (x,y-1,z), (x,y+1,z)]
        elseif dimension == 3
            potential_neighbors = [(x-1,y,z), (x+1,y,z), (x,y-1,z), (x,y+1,z), (x,y,z-1), (x,y,z+1)]
        end
        real_neighbors = Int[]
        for (x,y,z) in potential_neighbors

            if 1<=x<=box_width && 1<=y<=box_width && 1<=z<=box_width
                push!(real_neighbors, coordinates_to_node(x,y,z,box_width))
            end
        end
        push!(connectivity_matrix, real_neighbors)
    end
    return connectivity_matrix
end

"given a spatial index, get (node index, original species index)."
function from_spatial_spec(ind, num_species)
    fldmod1(ind, num_species)
end

"get the sptial index of the species in node"
function to_spatial_spec(node, ind, num_species)
    return (node-1)*num_species + ind
end
