using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments.db", "hpc/results", true)

config = Dict{Symbol, Any}(
    :nbits => IterableVariable(collect(4:2:12)),
    :nlayers => IterableVariable(collect(2:2:80)),
    :nrepeats => 10000,
    :gate_index => 1,
    :J => 1.0,
    :h => 0.5,
    :g => 0.0
)

experiment = Experiment(
    name="Barren Plateau Vanilla",
    include_file="hpc/barren_plateau_trial.jl",
    function_name="run_trial",
    configuration = deepcopy(config)
)

@execute experiment db DistributedMode