using Plots

"return coordinates from node index and number of sites per box edge"
node_to_coordinates(j,m) = ( (j-1)%m+1,(div(j-1,m))%m+1,(div(j-1,m^2)+1) )

"return node index from coordinates and number of sites per box edge"
coordinates_to_node(x,y,z,m) = x + (y-1)*m + (z-1)*m^2

"return node index from coordinates and number of sites per box edge"
coordinates_to_node(x,y,m) = x + (y-1)*m

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

"get frame k"
function get_frame(k, sol, num_species, num_sites_per_edge, labels, title)
    times = sol.t
    states = sol.u
    h = 1/num_sites_per_edge
    t = times[k]
    state = states[k]
    plt = plot(xlim=(0,1), ylim=(0,1), title = "$title, $(round(t, sigdigits=3)) seconds")

    species_seriess_x = [[] for i in 1:num_species]
    species_seriess_y = [[] for i in 1:num_species]
    for (species_spatial_index, number_of_molecules) in enumerate(state)
        node, species = from_spatial_spec(species_spatial_index, num_species)
        x,y,_ = node_to_coordinates(node, num_sites_per_edge)
        for k in 1:number_of_molecules
            push!(species_seriess_x[species], x*h - h/2 + h*rand())
            push!(species_seriess_y[species], y*h - h/2 + h*rand())
        end
    end
    for species in 1:num_species
        scatter!(plt, species_seriess_x[species], species_seriess_y[species], label = labels[species], marker = 3)
    end
    xticks!(plt, range(0,1,length = num_sites_per_edge+1))
    yticks!(plt, range(0,1,length = num_sites_per_edge+1))
    xgrid!(plt, 1, 0.7)
    ygrid!(plt, 1, 0.7)
    return plt
end

"make an animation of solution sol in 2 dimensions"
function animate_2d(sol, num_sites_per_edge; species_labels, title, verbose = true)
    num_frames = length(sol.t)
    num_species = Integer(length(sol.u[1])/num_sites_per_edge^2)
    anim = @animate for k=1:num_frames
        verbose && println("Making frame $k")
        get_frame(k, sol, num_species, num_sites_per_edge, species_labels, title)
    end
    anim
end
