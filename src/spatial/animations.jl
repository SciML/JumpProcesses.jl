"""
A file for animations of RDME
"""
using Plots

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
