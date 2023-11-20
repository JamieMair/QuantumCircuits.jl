using Revise
using ProgressBars;
using Statistics;
using CairoMakie;
using LaTeXStrings;
include("test_brickwork_problem.jl")

function gate_gradient(nbits, nlayers, H, ψ₀, index=1)
    circuit = GenericBrickworkCircuit(nbits, nlayers)
    Random.rand!(circuit.gate_angles)
    circuit.gate_angles .*= 2 * π
    return gradient(H, ψ₀, circuit, index)
end

function experiment(nbits, nlayers, nrepeats, hamiltonian_params; use_progress=true, index=1)
    # Convert to arrays
    nbits = typeof(nbits) <: Number ? [nbits] : nbits
    nlayers = typeof(nlayers) <: Number ? [nlayers] : nlayers

    iter = Iterators.product(1:nrepeats, enumerate(nbits), enumerate(nlayers))
    results = Array{Float64}(undef, size(iter)...)
    iter = use_progress ? ProgressBar(iter) : iter

    initial_ψs = map(nbits) do nb
        return zero_state_tensor(nb)
    end
    Hs = map(nbits) do nb
        H = build_hamiltonian(nb, hamiltonian_params...);
        return H
    end

    # Convert to arrays
    nbits = typeof(nbits) <: Number ? [nbits] : nbits
    nlayers = typeof(nlayers) <: Number ? [nlayers] : nlayers
    for (r, (i, nb), (j, nl)) in iter
        H = Hs[i]
        ψ₀ = initial_ψs[i]
        results[r, i, j] = gate_gradient(nb, nl, H, ψ₀, index)
    end
    return results;
end

function layer_plot_title(hps)
    return "TFIM " * join(("$(k)=$(v)" for (k,v) in pairs(hps)), ", ")
end

function plot_var_grads_vs_layers(nbits, nlayers, hamiltonian_params, results)
    # Convert to arrays
    nbits = typeof(nbits) <: Number ? [nbits] : nbits
    nlayers = typeof(nlayers) <: Number ? [nlayers] : nlayers

    f = Figure()
    ax = Axis(f[1,1],
        title=layer_plot_title(hamiltonian_params),
        xlabel="# layers",
        ylabel=L"\text{var}\left ([\nabla E]_1 \right )",
        yscale=log10)

    for (i, nb) in enumerate(nbits)
        layer_grads = view(results, :, i, :)
        var_grads = reshape(var(layer_grads, dims=1), :)
        scatter!(ax, nlayers, var_grads, label="n=$(nb)")
    end

    f[1,2] = Legend(f, ax, "# Sites", framevisible=false)

    return f
end

hamiltonian_params = (;
    J=1,
    h=0.5,
    g=0)
nbits = 10;
nlayers = 2:2:40;
nrepeats = 5000;


results = experiment(nbits, nlayers, nrepeats, hamiltonian_params)
plot_var_grads_vs_layers(nbits, nlayers, hamiltonian_params, results)


# Get results from HPC
using Experimenter
import Serialization
using DataFrames
using SQLite
db_path = joinpath(@__DIR__, "hpc/results/barren_plateau_experiments.db");
file_name = splitpath(db_path)[end];
db = open_db(file_name, dirname(db_path));
function _deserialize_columns(df::DataFrame)
    for colname in names(df)
        col = df[!, colname]
        if eltype(col) <: Vector{UInt8} # raw binary
            df[!, colname] = map(col) do c
                io = IOBuffer(c)
                return (Serialization.deserialize(io)).object
            end
        elseif eltype(col) <: SQLite.Serialized
            df[!, colname] = map(x->x.object, col)
        end
    end
    return df
end
function custom_get_trials(db::ExperimentDatabase, name)
    sql = raw"""
    SELECT name, Trials.id as id, experiment_id, Trials.configuration as configuration, results, trial_index, has_finished 
    FROM Trials 
    INNER JOIN Experiments ON Experiments.id == Trials.experiment_id 
    WHERE name = ? 
    ORDER BY trial_index
    """
    df = _deserialize_columns(SQLite.DBInterface.execute(db._db, sql, (name,)) |> DataFrame)
    return [Trial(row) for row in eachrow(df)]
end
trials = custom_get_trials(db, "Barren Plateau Vanilla");

nbits = sort([Set([t.configuration[:nbits] for t in trials])...])
nlayers = sort([Set([t.configuration[:nlayers] for t in trials])...])

hamiltonian_params = begin
    J = sort([Set([t.configuration[:J] for t in trials])...])
    h = sort([Set([t.configuration[:h] for t in trials])...])
    g = sort([Set([t.configuration[:g] for t in trials])...])
    @assert length(J) == 1
    @assert length(h) == 1
    @assert length(g) == 1
    hamiltonian_params = (;
    J=J[begin],
    h=h[begin],
    g=g[begin])
    return hamiltonian_params
end

begin
    gate_index = sort([Set([t.configuration[:gate_index] for t in trials])...])
    @assert length(gate_index) == 1
    gate_index = gate_index[begin]
end

begin
    nrepeats = sort([Set([t.configuration[:nrepeats] for t in trials])...])
    @assert length(nrepeats) == 1
    nrepeats = nrepeats[begin]
end

results = begin
    results = zeros(Float64, (nrepeats, length(nbits), length(nlayers)))

    for ((i, nb), (j, nl)) in Iterators.product(enumerate(nbits), enumerate(nlayers))
        t = first([t for t in trials if t.configuration[:nlayers] == nl && t.configuration[:nbits] == nb])
        results[:, i, j] .= t.results[:gradients]
    end
    return results
end

plot_var_grads_vs_layers(nbits, nlayers, hamiltonian_params, results)