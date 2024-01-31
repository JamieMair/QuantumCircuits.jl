using Experimenter
using Experimenter.DataFrames
using CairoMakie

db = open_db("experiments.db", "hpc/results", true)
original_results = (
    DataFrame(get_trials_by_name(db, "NN Optimisation Low LR")),
    DataFrame(get_trials_by_name(db, "NN Optimisation Test Large NN 3"))  # 1250 neuron
)
results = sort(process_results(original_results...), by=x -> sum(y -> y.neurons, x.c_architecture[1]))

f = plot_all_energy_trajectories(results...)

CairoMakie.save("hpc/results/nn_optimisation_low_lr.pdf", f; pt_per_unit=1)