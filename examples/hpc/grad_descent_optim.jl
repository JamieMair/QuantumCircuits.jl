using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments.db", "hpc/results", true)

nrepeats = 10
config = Dict{Symbol,Any}(
    :nbits => IterableVariable(collect(4:2:8)),
    :nlayers => IterableVariable([4, 8, 12, 16]),
    :repeat_id => IterableVariable(collect(1:nrepeats)),
    :J => 1.0,
    :h => 0.5,
    :g => 0.0,
    :learning_rate => IterableVariable([0.0001]),
    :use_gpu => false,
    :architecture => [],
    :epochs => 150,
)

experiment = Experiment(
    name="Gradient Optimisation Low LR",
    include_file="hpc/nn_optim_trial.jl",
    function_name="run_trial",
    configuration=deepcopy(config)
)

Experimenter.Cluster.init()

@execute experiment db DistributedMode