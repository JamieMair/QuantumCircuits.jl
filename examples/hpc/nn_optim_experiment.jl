using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments.db", "hpc/results", true)

nrepeats = 10
config = Dict{Symbol,Any}(
    :nbits => IterableVariable(collect(4:2:12)),
    :nlayers => IterableVariable([4, 8, 12, 16, 20, 24, 28, 32, 36, 40]),
    :repeat_id => IterableVariable(collect(1:nrepeats)),
    :J => 1.0,
    :h => 0.5,
    :g => 0.0,
    :use_gpu => true,
    :architecture => [
        (; neurons=50, activation=:tanh),
        (; neurons = 50, activation = :tanh),
        (; neurons=50, activation=:tanh)
    ],
    :epochs => 100,
)

experiment = Experiment(
    name="Test NN optim",
    include_file="hpc/nn_optim_trial.jl",
    function_name="run_trial",
    configuration=deepcopy(config)
)

Experimenter.Cluster.init()

@execute experiment db DistributedMode