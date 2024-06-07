using Revise
using ProgressBars;
using Statistics;
using CairoMakie;
using LaTeXStrings;
using Experimenter
import Serialization
using DataFrames
using SQLite

cd(joinpath(@__DIR__, ".."))
include("utils.jl")

db_path = abspath(joinpath(@__DIR__, "../hpc/results/tuning_experiments.db"));
file_name = splitpath(db_path)[end];
db = open_db(file_name, dirname(db_path));

original_results = DataFrame(get_trials_by_name(db, "TFIM NC Learning Rate Tuning 1"));
results = process_results(original_results; groupby_colnames=[:c_architecture, :c_learning_rate]);

learning_rate_tuples = [(i,results[i].c_learning_rate[begin]) for i in 1:5]
rate_indices = [y[1] for y in sort(learning_rate_tuples, by=x->x[2])]


best_foreach_architecture = map(1:4) do i
    arch_results = results[rate_indices .+ (i-1)*length(rate_indices)]
    lowest_energy_idx = argmin(minimum(minimum(es) for es in x.r_energy_trajectory) for x in arch_results)
    return arch_results[lowest_energy_idx]
end
f = plot_all_energy_trajectories(best_foreach_architecture...; plot_log=true)


fig_dir = abspath(joinpath(@__DIR__, "../figures"))
!isdir(fig_dir) && mkdir(fig_dir)
CairoMakie.save(joinpath(fig_dir, "optimised_learning_rates_compared.pdf"), f; pt_per_unit=1)