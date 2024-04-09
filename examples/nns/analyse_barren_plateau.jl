using Revise
using ProgressBars;
using Statistics;
using CairoMakie;
using LaTeXStrings;
using Experimenter
import Serialization
using DataFrames
using SQLite
include("utils.jl")

function layer_plot_title(hps)
    return "TFIM " * join(("$(k)=$(v)" for (k,v) in pairs(hps)), ", ")
end

db_path = abspath(joinpath(@__DIR__, "../hpc/results/experiments_plateaux.db"));
file_name = splitpath(db_path)[end];
db = open_db(file_name, dirname(db_path));

original_results = DataFrame(get_trials_by_name(db, "Barren Plateau NNs Full"));
dfs = process_results(original_results; groupby_colnames=[:c_architecture]);

f = plot_barren_plateaux_v2(dfs...; join_axes=false)
CairoMakie.save(abspath(joinpath(@__DIR__, "../hpc/results/barren_plateaux_all_architectures.pdf")), f; pt_per_unit=1)