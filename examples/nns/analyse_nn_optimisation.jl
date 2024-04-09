using Experimenter
using Experimenter.DataFrames
using CairoMakie
include("utils.jl")
cd(joinpath(@__DIR__, ".."))
db = open_db("experiments_long.db", "hpc/results", true)
original_results = (
    DataFrame(get_trials_by_name(db, "NN Optimisation Near Critical With Gradients")),
)
results = process_results(original_results...)
f = plot_all_energy_trajectories(results...; plot_log=true)
plot_durations(results...)

CairoMakie.save("hpc/results/nn_optimisation_non_integrable.pdf", f; pt_per_unit=1)
