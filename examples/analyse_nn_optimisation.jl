using Experimenter
using Experimenter.DataFrames
using Statistics
db = open_db("experiments.db", "hpc/results", true)
results = DataFrame(get_trials_by_name(db, "NN Optimisation 1"))

function unpack_cols!(df, col_name, prepend="")
    for k in keys(df[1, col_name])
        key_name = prepend * string(k)
        df[!, key_name] = map(df[!, col_name]) do dict
            if haskey(dict, k)
                return dict[k]
            else
                return missing
            end
        end
    end
end

unpack_cols!(results, :configuration, "c_")
unpack_cols!(results, :results, "r_")


using CairoMakie
using ColorSchemes
using LaTeXStrings

function plot_all_energy_trajectories(df)
    
    nbits_set = sort(unique(df[!, :c_nbits]))
    nlayers_set = reverse(sort(unique(df[!, :c_nlayers])))
    color_scheme = ColorSchemes.matter
    min_layers, max_layers = extrema(nlayers_set)
    m = log(256/1)/log(max_layers/min_layers)
    c = log(1) - m * log(min_layers)
    convert_col_to_idx(nl) = Int(257-round((1+ sqrt((nl-min_layers) / (max_layers-min_layers)) * 255)))
    color_map = Dict((map(nlayers_set) do nl
        index = convert_col_to_idx(nl)
        # i = max(1, min(256, Int(round(exp(m*log(nl)+c)))))
        return nl=>color_scheme[index]
    end)...)

    f = Figure(size=(800,3000))
    for (i, nbits) in enumerate(nbits_set)
        
        ax = Axis(f[i, 1], xlabel=L"t", ylabel=L"\langle H \rangle", title=LaTeXString("\$\$n=$nbits\$\$"))
        ges = unique(df[df[!, :c_nbits] .== nbits, :r_ground_energy])
        @assert length(ges) == 1 "There should be only one ground energy for a fixed number of qubits."

        for nlayers in nlayers_set
            trajectories = subset(df,
                :c_nbits => nb -> nb .== nbits,
                :c_nlayers => nl -> nl .== nlayers)[!, :r_energy_trajectory]
            c = color_map[nlayers]

            combined_trajs = hcat(trajectories...)
            mean_trajectory = reshape(mean(combined_trajs, dims=2), :)
            err_trajectory = reshape(std(combined_trajs, dims=2), :)
            epochs = 0:(length(mean_trajectory)-1)


            band_c = RGBAf(c.r, c.g, c.b, 0.2)
            band!(ax, epochs, mean_trajectory .- err_trajectory, mean_trajectory .+ err_trajectory; alpha=0.2, label=nothing, color=band_c)

            plot_trajectories!(ax, trajectories; alpha=0.1, label=nothing, color=c)
            
            lines!(ax, epochs, mean_trajectory; label="$nlayers", color=c, lw=3)      
            xlims!(ax, 0, maximum(epochs))
        end

        hlines!(ax, ges, label=L"E_0", linestyle=:dash, alpha=0.5, color=:black)

    end

    new_colour_scheme = ColorScheme(map(LinRange(min_layers, max_layers, 256)) do nl
        color_scheme[convert_col_to_idx(nl)]
    end)

    

    cbar = Colorbar(f[length(nbits_set)+1, 1], limits=(min_layers, max_layers), ticks=nlayers_set, colormap=new_colour_scheme, vertical=false, label="Layers")

    return f
end
function plot_trajectories!(ax, trajectories; kwargs...)
    for traj in trajectories
        lines!(ax, 0:(length(traj)-1), traj; kwargs...)
    end

    return ax
end