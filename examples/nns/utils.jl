using Statistics
using CairoMakie
using ColorSchemes
import Flux: destructure
using LaTeXStrings
include("../matrix_tfim.jl")

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

function unpack_results(dfs...; groupby_colnames=[:c_architecture, :c_learning_rate])
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


        for _subdf in groupby(sub_df, groupby_colnames)
            push!(results, _subdf)
        end
    end

    return [results...]
end
function process_results(dfs...; kwargs...)
    # Sort the dataframes by the number of neurons
    df_to_neuron_count(df) = length(df.c_architecture[1]) == 0 ? 0 : sum(y -> y.neurons, df.c_architecture[1])
    results = sort(unpack_results(dfs...; kwargs...), by=df_to_neuron_count)

    # Combine same architectures
    architectures = sort(unique([df.c_architecture[1] for df in results]), by=x -> sum(y -> y.neurons, x, init=0))
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

    ground_energy_fn = MemoisedGroundStateFn(Float64)
    

    # Map each row to a list of axes to align them later
    rows_axes = Dict{Int,Any}()

    for (j, df) in enumerate(dfs)

        nbits_local_set = Set(sort(unique(df[!, :c_nbits])))
        nlayers_local_set = reverse(sort(unique(df[!, :c_nlayers])))



        for (i, nbits) in enumerate(nbits_set)

            max_energy = -Inf

            if !(nbits in nbits_local_set)
                continue # Skip over graphs that don't exist
            end

            current_entries = df[!, :c_nbits] .== nbits
            archs = unique(df[current_entries, :c_architecture])
            @assert length(archs) == 1 "There should be only be a single architecture."

            nparams = length(destructure(df[current_entries, :r_model_state][1])[1])
            nparams_log10 = Int(ceil(log10(nparams)))
            additional_args = plot_log ? Dict{Symbol,Any}(
                # :xscale => log10,
                :yscale => log10,
            ) : Dict{Symbol,Any}()
            xlabel = i < length(nbits_set) ? "" : L"t"
            ylabel = j > 1 ? "" : L"\frac{\langle H \rangle - H_0}{H_0}"
            ax = Axis(f[i, j]; xlabel, ylabel, title=LaTeXString("\$n=$nbits, |\\theta|\\approx 10^{$nparams_log10}\$"), additional_args...)

            if haskey(rows_axes, i)
                push!(rows_axes[i], ax)
            else
                rows_axes[i] = [ax]
            end

            J = first(unique(df[current_entries, :c_J]))
            h = first(unique(df[current_entries, :c_h]))
            g = first(unique(df[current_entries, :c_g]))

            ground_energy, _ = ground_energy_fn(nbits, J, h, g)
            # hlines!(ax, ground_energy, label=L"E_0", linestyle=:dash, alpha=0.5, color=:black)

            for nlayers in nlayers_local_set
                trajectories = map(subset(df,
                    :c_nbits => nb -> nb .== nbits,
                    :c_nlayers => nl -> nl .== nlayers)[!, :r_energy_trajectory]) do trajectory

                        return (trajectory .- ground_energy) ./ abs(ground_energy)
                end
                c = color_map[nlayers]


                combined_trajs = hcat(trajectories...)
                if length(combined_trajs) == 0
                    continue
                end
                mean_trajectory = reshape(mean(combined_trajs, dims=2), :)
                err_trajectory = reshape(std(combined_trajs, dims=2), :)
                epochs = 0:(length(mean_trajectory)-1)

                max_epoch = argmax(mean_trajectory)

                max_energy_layer = mean_trajectory[max_epoch] + err_trajectory[max_epoch]
                if max_energy_layer > max_energy
                    max_energy = max_energy_layer
                end


                band_c = RGBAf(c.r, c.g, c.b, 0.2)
                # band!(ax, epochs, mean_trajectory .- err_trajectory, mean_trajectory .+ err_trajectory; alpha=0.2, label=nothing, color=band_c)

                plot_trajectories!(ax, trajectories; alpha=0.025, label=nothing, color=c)

                lines!(ax, epochs, mean_trajectory; label="$nlayers", color=c, lw=3)
                
                xlims!(ax, 0, maximum(epochs))
            end

            # ylims!(ax, 0.0000000001, max_energy)
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

function plot_barren_plateaux(dfs...; plot_log=true)

    nbits_set = sort(unique(vcat((unique(df[!, :c_nbits]) for df in dfs)...)))

    f = Figure(size=(300*length(dfs), 400))

    nlayers_set = sort(unique(vcat((unique(df[!, :c_nlayers]) for df in dfs)...)))
    nbits_set = sort(unique(vcat((unique(df[!, :c_nbits]) for df in dfs)...)))
    min_nbits, max_nbits = extrema(vcat(([extrema(df[!, :c_nbits])...] for df in dfs)...))
    m = log(256 / 1) / log(max_nbits / min_nbits)
    c = log(1) - m * log(min_nbits)
    convert_col_to_idx(nl) = Int(257 - round((1 + sqrt((nl - min_nbits) / (max_nbits - min_nbits)) * 255)))
    color_scheme = ColorSchemes.matter

    color_map = Dict((
        map(nbits_set) do nb
            index = convert_col_to_idx(nb)
            # i = max(1, min(256, Int(round(exp(m*log(nl)+c)))))
            return nb => color_scheme[index]
        end
    )...)

    symbol_list = [
        :rect,
        :circle,
        :cross,
        :utriangle
    ]
    linestyle_list = [
        :dash,
        :dashdot,
        :dashdotdot,
        :solid,
    ]

    labels = [
        "No NN",
        "3 x 50",
        "3 x 250",
        "3 x 1250",
    ]

    graph_elements = map([a for a in zip(symbol_list, linestyle_list)]) do (marker, linestyle)
        return [LineElement(color = :black, linestyle = linestyle),
        MarkerElement(color = :black, marker = marker, strokecolor = :black)]
    end


    row_axs = []

    for (j, df) in enumerate(dfs)

        nbits_local_set = Set(sort(unique(df[!, :c_nbits])))
        nlayers_local_set = sort(unique(df[!, :c_nlayers]))

        additional_args = plot_log ? Dict{Symbol,Any}(
            # :xscale => CairoMakie.Makie.pseudolog10,
            :yscale => log10,
        ) : Dict{Symbol,Any}()


        ylabel = j > 1 ? "" : L"\mathbb{E} \left [ \frac{ \left {||} \overline{\nabla_\theta} \right {||} }{N_\text{params}} \right ]"
        ax = Axis(f[1, j]; xlabel=L"l", ylabel, title=labels[j], yscale=log10)
        push!(row_axs, ax)

        for (i, nbits) in enumerate(nbits_set)

            if !(nbits in nbits_local_set)
                continue # Skip over graphs that don't exist
            end

            current_entries = df[!, :c_nbits] .== nbits
            archs = unique(df[current_entries, :c_architecture])
            @assert length(archs) == 1 "There should be only be a single architecture."



            mean_gradient_vector_size = map(nlayers_local_set) do nlayers
                tmp_df = subset(df,
                    :c_nbits => nb -> nb .== nbits,
                    :c_nlayers => nl -> nl .== nlayers)

                grads = map(tmp_df[!, :r_training_info]) do info
                    k = info[:gradient_info][1]
                    nparams = QuantumCircuits.brickwork_num_gates(nbits, nlayers) * 15
                    return k.norm_grads / nparams
                end

                if length(grads) == 0
                    return missing
                else
                    return mean(grads)
                end
            end

            c = color_map[nbits]

            linestyle = linestyle_list[j]

            lines!(ax, nlayers_local_set, mean_gradient_vector_size; label="$nbits", color=c, linestyle=linestyle, alpha=0.4)

            marker = symbol_list[j]

            scatter!(ax, nlayers_local_set, mean_gradient_vector_size; label="$nbits", color=c, marker=marker)
        end

    end

    new_colour_scheme = ColorScheme(map(LinRange(min_nbits, max_nbits, 256)) do nl
        color_scheme[convert_col_to_idx(nl)]
    end)

    # Legend(f[1,2], graph_elements, labels)
    linkyaxes!(row_axs...)

    cbar = Colorbar(f[2, 1:length(dfs)], limits=(min_nbits, max_nbits), ticks=nbits_set, colormap=new_colour_scheme, vertical=false, label="N")

    return f
end


function count_neurons(df)
    return length(df.c_architecture[1]) == 0 ? 0 : sum(y -> y.neurons, df.c_architecture[1])
end

function plot_durations(dfs...; plot_log=false)
    nbits_set = sort(unique(vcat((unique(df[!, :c_nbits]) for df in dfs)...)))

    f = Figure(size=(400 * length(dfs), 400 + 100))

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
    rows_axes = Dict{Int,Any}()


    for (j, df) in enumerate(dfs)

        nbits_local_set = Set(sort(unique(df[!, :c_nbits])))
        nlayers_local_set = reverse(sort(unique(df[!, :c_nlayers])))

        nparams = length(destructure(df[!, :r_model_state][1])[1])
        nparams_log10 = Int(ceil(log10(nparams)))
        additional_args = plot_log ? Dict{Symbol,Any}(
            :xscale => CairoMakie.Makie.pseudolog10,
            :yscale => CairoMakie.Makie.pseudolog10,
        ) : Dict{Symbol,Any}()
        ax = Axis(f[1, j], xlabel=L"N", ylabel=L"D(s)", title=LaTeXString("\$|\\theta|\\approx 10^{$nparams_log10}\$"); additional_args..., :yscale => log2, :xscale => log2)

        if haskey(rows_axes, j)
            push!(rows_axes[j], ax)
        else
            rows_axes[j] = [ax]
        end

        for nlayers in nlayers_local_set
            nbits_local = sort(collect(nbits_local_set))
            mean_durations = map(nbits_local) do nbits
                a = df[(df[!, :c_nbits].==nbits).&&(df[!, :c_nlayers].==nlayers), :r_duration_s]
                if length(a) == 0
                    return missing
                end
                return mean(a)
            end
            error_durations = map(nbits_local) do nbits
                a = df[(df[!, :c_nbits].==nbits).&&(df[!, :c_nlayers].==nlayers), :r_duration_s]
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

struct MemoisedGroundStateFn{T} <: Function
    answers::Dict{Tuple{Int,T,T,T},Tuple{T,Vector{T}}}
end

function MemoisedGroundStateFn(T)
    return MemoisedGroundStateFn{T}(Dict{Tuple{Int,T,T,T},Tuple{T,Vector{T}}}())
end

function (fn::MemoisedGroundStateFn{T})(n::Int, J, h, g) where {T}
    J = convert(T, J)
    h = convert(T, h)
    g = convert(T, g)

    fn_args = (n, J, h, g)

    if haskey(fn.answers, fn_args)
        return fn.answers[fn_args]
    else
        ans = find_tfim_ground_state(fn_args...)
        fn.answers[fn_args] = ans
        return ans
    end
end