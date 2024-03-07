using Experimenter
using Experimenter.DataFrames
using CairoMakie
include("utils.jl")
cd(joinpath(@__DIR__, ".."))
db = open_db("experiments.db", "hpc/results", true)
original_results = (
    DataFrame(get_trials_by_name(db, "NN Optimisation J=h")),
    DataFrame(get_trials_by_name(db, "NN Optimisation J=h (increased layers)")),
    DataFrame(get_trials_by_name(db, "NN Optimisation J=h (high qubits)")),
    # DataFrame(get_trials_by_name(db, "NN Optimisation Test Large NN 3")),  # 1250 neuron
    # DataFrame(get_trials_by_name(db, "Gradient Optimisation Low LR")),  # 1 neuron (i.e. no neural network)
)
results = process_results(original_results...)
f = plot_all_energy_trajectories(results...)
plot_durations(results...)

CairoMakie.save("hpc/results/nn_optimisation_crtical_point.pdf", f; pt_per_unit=1)

