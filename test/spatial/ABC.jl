using DiffEqJump, DiffEqBase, Plots

function get_connectivity_list(box_width :: Integer, dimension :: Integer)
    @assert 1 <= dimension <= 3
    vol_num = box_width^dimension
    connectivity_matrix = Array{Array{Int64,1},1}(undef, 0)
    for j in 1:vol_num
        x,y,z = phi(j,box_width)
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
                push!(real_neighbors, phi_inverse(x,y,z,box_width))
            end
        end
        push!(connectivity_matrix, real_neighbors)
    end
    return connectivity_matrix
end
phi(j,m) = ( (j-1)%m+1,(div(j-1,m))%m+1,(div(j-1,m^2)+1) )
phi_inverse(x,y,z,m) = x + (y-1)*m + (z-1)*m^2

# ABC model A + B <--> C
reactstoch = [
    [1 => 1, 2 => 1],
    [3 => 1],
]
netstoch = [
    [1 => -1, 2 => -1, 3 => 1],
    [1 => 1, 2 => 1, 3 => -1]
]
spec_to_dep_jumps = [[1],[1],[2]]
jump_to_dep_specs = [[1,2,3],[1,2,3]]
rates = [10., 100.]
majumps = MassActionJump(rates, reactstoch, netstoch)
prob = DiscreteProblem([500,500,0],(0.0,2.0), rates)
jump_prob = JumpProblem(prob, RSSACR(), majumps, save_positions=(false,false), vartojumps_map=spec_to_dep_jumps, jumptovars_map=jump_to_dep_specs)

# Graph setup
domain_size = 1.0 #μ-meter
num_sites_per_edge = 5
diffusivity = 10.0
hopping_rate = diffusivity * (num_sites_per_edge/domain_size)^2
connectivity_list = get_connectivity_list(num_sites_per_edge, 3)
num_nodes = length(connectivity_list)

diff_rates_for_edge = [hopping_rate for species in 1:length(jump_prob.prob.u0)]
diff_rates = [[diff_rates_for_edge for j in 1:length(connectivity_list[i])] for i in 1:num_nodes]

# Solving
alg = NRM()
println("Solving with $alg")
spatial_jump_prob = to_spatial_jump_prob(connectivity_list, diff_rates, jump_prob, alg)
sol = solve(spatial_jump_prob, SSAStepper(), saveat = prob.tspan[2]/100)
println("Plotting")
labels = vcat([["A $i", "B $i", "C $i"] for i in 1:num_nodes]...)
trajectories = [hcat(sol.u...)[i,:] for i in 1:length(spatial_jump_prob.prob.u0)]
plot1 = plot(sol.t, trajectories[1], label = labels[1])
for i in 2:3
    plot!(plot1, sol.t, trajectories[i], label = labels[i])
end
title!("A + B <--> C RDME")
xaxis!("time")
yaxis!("number")

# Make animation
"get frame k"
function get_frame(k,sol,num_sites_per_edge)
    times = sol.t
    states = sol.u
    d = 1/num_sites_per_edge
    t = times[k]
    state = states[k]
    title = "A + B <--> C, $t"
    plt = plot(xlim=(0,1), ylim=(0,1), title = title)
    species_locations = Arra
    protein_series = Tuple{Float64,Float64}[]
    mrna_series = Tuple{Float64,Float64}[]
    for j in 1:length(state)
        x,y,_ = SpatialSSAs.phi(j,m)./m
        j_state = state[j]
        for i in 1:j_state[3]
            push!(protein_series, (x - d/2 + d*rand(), y - d/2 + d*rand()))
        end
        for i in 1:j_state[2]
            push!(mrna_series, (x - d/2 + d*rand(), y - d/2 + d*rand()))
        end
    end
    x,y,_ = SpatialSSAs.phi(center,m)./m

    scatter!(plt, protein_series, color=:"red",label=:"Protein",marker=1)
    scatter!(plt, mrna_series, color=:"green",label=:"mRNA",marker=3)
    xticks!(plt, range(0,1,length=m))
    yticks!(plt, range(0,1,length=m))
    xgrid!(plt, 1, 0.7)
    ygrid!(plt, 1, 0.7)
    if state[center][1] != 0
        scatter!(plt, (x,y), color=:"blue",label=:"DNA",marker=8)
    else scatter!(plt, (2.0, 2.0), color=:"blue", label=:"DNA", marker=8) end

    if state[center][4] != 0
        scatter!(plt, (x, y), color=:"black",label=:"repressed DNA",marker=8)
    else scatter!(plt, (2.0, 2.0), color=:"black",label=:"repressed DNA",marker=8) end
    return plt
end

@gif for k=1:N
    get_frame(k,states,times,m)
    println("Done with frame $k")
end
