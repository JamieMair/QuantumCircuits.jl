using Experimenter
using Experimenter.DataFrames
using CairoMakie
include("utils.jl")
cd(joinpath(@__DIR__, ".."))
db = open_db("experiments.db", "hpc/results", true)
original_results = (
    DataFrame(get_trials_by_name(db, "NN Optimisation Low LR")), # 50 and 250 neurons
    DataFrame(get_trials_by_name(db, "NN Optimisation Test Large NN 3")),  # 1250 neuron
    DataFrame(get_trials_by_name(db, "Gradient Optimisation Low LR")),  # 1 neuron (i.e. no neural network)
)
results = sort(process_results(original_results...), by=x -> length(x.c_architecture[1]) == 0 ? 0 : sum(y -> y.neurons, x.c_architecture[1]))

f = plot_all_energy_trajectories(results...; plot)

CairoMakie.save("hpc/results/nn_optimisation_low_lr.pdf", f; pt_per_unit=1)