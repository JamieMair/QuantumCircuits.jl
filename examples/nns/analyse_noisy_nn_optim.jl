using Experimenter
using Experimenter.DataFrames
using CairoMakie
include("utils.jl")
cd(joinpath(@__DIR__, ".."))
db = open_db("experiments_noisy.db", "hpc/results", true)
original_results = (
    DataFrame(get_trials_by_name(db, "NN Noisy Optim Longer Test")),
)
results = process_results(original_results...; groupby_colnames=[:c_architecture, :c_hamiltonian_noise], should_merge_architectures=false)

f = plot_all_energy_trajectories(results[1], results[4]; plot_log=true)

CairoMakie.save("hpc/results/nn_noisy_optimisation.pdf", f; pt_per_unit=1)
