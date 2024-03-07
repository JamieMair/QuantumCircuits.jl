using Statistics
using CairoMakie
using ColorSchemes
import Flux: destructure
using LaTeXStrings

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

function unpack_results(dfs...)
    results = []
    for df in dfs
        sub_df = df[df[!, :has_finished], :]
        unpack_cols!(sub_df, :configuration, "c_")
        unpack_cols!(sub_df, :results, "r_")

        if !hasproperty(sub_df, :r_ground_state)
            sub_df[!, :r_ground_state] = [missing for _ in 1:length(sub_df.id)]
        end
        if !hasproperty(sub_df, :r_ground_energy)
            sub_df[!, :r_ground_energy] = [missing for _ in 1:length(sub_df.id)]
        end


        for _subdf in groupby(sub_df, [:c_architecture, :c_learning_rate])
            push!(results, _subdf)
        end
    end

    return [results...]
end
function process_results(dfs...)
    # Sort the dataframes by the number of neurons
    df_to_neuron_count(df) = length(df.c_architecture[1]) == 0 ? 0 : sum(y -> y.neurons, df.c_architecture[1])
    results = sort(unpack_results(dfs...), by=df_to_neuron_count)

    # Combine same architectures
    architectures = sort(unique([df.c_architecture[1] for df in results]), by=x->sum(y->y.neurons, x, init=0))
    result_groups = [[df for df in results if df.c_architecture[1] == c] for c in architectures]
    results = [vcat(rs...) for rs in result_groups]

    # Final sort to make sure
    results = sort(results, by=df_to_neuron_count)
    return results
end

function plot_all_energy_trajectories(dfs...; plot_log=false)
    nbits_set = sort(unique(vcat((unique(df[!, :c_nbits]) for df in dfs)...)))

    f = Figure(size=(300 * length(dfs), 300 * length(nbits_set)))

    nlayers_set = sort(unique(vcat((unique(df[!, :c_nlayers]) for df in dfs)...)))
    min_layers, max_layers = extrema(vcat(([extrema(df[!, :c_nlayers])...] for df in dfs)...))
    m = log(256 / 1) / log(max_layers / min_layers)
    c = log(1) - m * log(min_layers)
    convert_col_to_idx(nl) = Int(257 - round((1 + sqrt((nl - min_layers) / (max_layers - min_layers)) * 255)))
    color_scheme = ColorSchemes.matter

    color_map = Dict((
        map(nlayers_set) do nl
            index = convert_col_to_idx(nl)
            # i = max(1, min(256, Int(round(exp(m*log(nl)+c)))))
            return nl => color_scheme[index]
        end
    )...)


    # Map each row to a list of axes to align them later
    rows_axes = Dict{Int, Any}()

    for (j, df) in enumerate(dfs)

        nbits_local_set = Set(sort(unique(df[!, :c_nbits])))
        nlayers_local_set = reverse(sort(unique(df[!, :c_nlayers])))



        for (i, nbits) in enumerate(nbits_set)

            if !(nbits in nbits_local_set)
                continue # Skip over graphs that don't exist
            end

            current_entries = df[!, :c_nbits].==nbits
            archs = unique(df[current_entries, :c_architecture])
            @assert length(archs) == 1 "There should be only be a single architecture."

            nparams = length(destructure(df[current_entries, :r_model_state][1])[1])
            nparams_log10 = Int(ceil(log10(nparams)))
            additional_args = plot_log ? Dict{Symbol, Any}(
                :xscale => CairoMakie.Makie.pseudolog10,
                :yscale => CairoMakie.Makie.pseudolog10,
            ) : Dict{Symbol, Any}()
            ax = Axis(f[i, j], xlabel=L"t", ylabel=L"\langle H \rangle", title=LaTeXString("\$n=$nbits, |\\theta|\\approx 10^{$nparams_log10}\$"); additional_args...)

            if haskey(rows_axes, i)
                push!(rows_axes[i], ax)
            else
                rows_axes[i] = [ax]
            end

            for nlayers in nlayers_local_set
                trajectories = subset(df,
                    :c_nbits => nb -> nb .== nbits,
                    :c_nlayers => nl -> nl .== nlayers)[!, :r_energy_trajectory]
                c = color_map[nlayers]


                combined_trajs = hcat(trajectories...)
                if length(combined_trajs) == 0
                    continue
                end
                mean_trajectory = reshape(mean(combined_trajs, dims=2), :)
                err_trajectory = reshape(std(combined_trajs, dims=2), :)
                epochs = 0:(length(mean_trajectory)-1)


                band_c = RGBAf(c.r, c.g, c.b, 0.2)
                band!(ax, epochs, mean_trajectory .- err_trajectory, mean_trajectory .+ err_trajectory; alpha=0.2, label=nothing, color=band_c)

                plot_trajectories!(ax, trajectories; alpha=0.1, label=nothing, color=c)

                lines!(ax, epochs, mean_trajectory; label="$nlayers", color=c, lw=3)
                xlims!(ax, 0, maximum(epochs))
            end

            if :r_ground_energy in propertynames(df)
                ges = unique(skipmissing(df[df[!, :c_nbits].==nbits, :r_ground_energy]))
                if length(ges) > 0
                    if length(ges) > 1 
                        @warn "There should be only one ground energy for a fixed number of qubits, but found $(length(ges)) for $(nbits)."
                    end
                    hlines!(ax, ges, label=L"E_0", linestyle=:dash, alpha=0.5, color=:black)
                end
            end

        end

    end
    new_colour_scheme = ColorScheme(map(LinRange(min_layers, max_layers, 256)) do nl
        color_scheme[convert_col_to_idx(nl)]
    end)

    for row in keys(rows_axes)
        linkyaxes!(rows_axes[row]...)
    end


    cbar = Colorbar(f[length(nbits_set)+1, 1:length(dfs)], limits=(min_layers, max_layers), ticks=nlayers_set, colormap=new_colour_scheme, vertical=false, label="Layers")

    return f
end


function count_neurons(df)
    return length(df.c_architecture[1]) == 0 ? 0 : sum(y -> y.neurons, df.c_architecture[1])
end

function plot_durations(dfs...; plot_log=false)
    nbits_set = sort(unique(vcat((unique(df[!, :c_nbits]) for df in dfs)...)))

    f = Figure(size=(400 * length(dfs), 400  + 100))

    nlayers_set = sort(unique(vcat((unique(df[!, :c_nlayers]) for df in dfs)...)))
    min_layers, max_layers = extrema(vcat(([extrema(df[!, :c_nlayers])...] for df in dfs)...))
    m = log(256 / 1) / log(max_layers / min_layers)
    c = log(1) - m * log(min_layers)
    convert_col_to_idx(nl) = Int(257 - round((1 + sqrt((nl - min_layers) / (max_layers - min_layers)) * 255)))
    color_scheme = ColorSchemes.matter

    color_map = Dict((
        map(nlayers_set) do nl
            index = convert_col_to_idx(nl)
            # i = max(1, min(256, Int(round(exp(m*log(nl)+c)))))
            return nl => color_scheme[index]
        end
    )...)

    # Map each row to a list of axes to align them later
    rows_axes = Dict{Int, Any}()


    for (j, df) in enumerate(dfs)

        nbits_local_set = Set(sort(unique(df[!, :c_nbits])))
        nlayers_local_set = reverse(sort(unique(df[!, :c_nlayers])))

        nparams = length(destructure(df[!, :r_model_state][1])[1])
        nparams_log10 = Int(ceil(log10(nparams)))
        additional_args = plot_log ? Dict{Symbol, Any}(
            :xscale => CairoMakie.Makie.pseudolog10,
            :yscale => CairoMakie.Makie.pseudolog10,
        ) : Dict{Symbol, Any}()
        ax = Axis(f[1, j], xlabel=L"N", ylabel=L"D(s)", title=LaTeXString("\$|\\theta|\\approx 10^{$nparams_log10}\$"); additional_args..., :yscale => log2, :xscale => log2)
        
        if haskey(rows_axes, j)
            push!(rows_axes[j], ax)
        else
            rows_axes[j] = [ax]
        end

        for nlayers in nlayers_local_set
            nbits_local = sort(collect(nbits_local_set))
            mean_durations = map(nbits_local) do nbits
                a = df[(df[!, :c_nbits].==nbits) .&& (df[!, :c_nlayers].==nlayers), :r_duration_s]
                if length(a) == 0
                    return missing
                end
                return mean(a)
            end
            error_durations = map(nbits_local) do nbits
                a = df[(df[!, :c_nbits].==nbits) .&& (df[!, :c_nlayers].==nlayers), :r_duration_s]
                if length(a) == 0
                    return missing
                end
                return std(a) / sqrt(length(a))
            end

            mask = (!ismissing).(mean_durations)
            nbits_local = nbits_local[mask]
            mean_durations = mean_durations[mask]
            error_durations = error_durations[mask]

            c = color_map[nlayers]
            CairoMakie.errorbars!(ax, nbits_local, mean_durations, error_durations, color=:black)
            CairoMakie.scatter!(ax, nbits_local, mean_durations, color=c)
        end
    end
    new_colour_scheme = ColorScheme(map(LinRange(min_layers, max_layers, 256)) do nl
        color_scheme[convert_col_to_idx(nl)]
    end)

    for row in keys(rows_axes)
        linkyaxes!(rows_axes[row]...)
    end
    
    cbar = Colorbar(f[2, 1:length(dfs)], limits=(min_layers, max_layers), ticks=nlayers_set, colormap=new_colour_scheme, vertical=false, label="Layers")

    return f
end

function plot_trajectories!(ax, trajectories; kwargs...)
    for traj in trajectories
        lines!(ax, 0:(length(traj)-1), traj; kwargs...)
    end

    return ax
end