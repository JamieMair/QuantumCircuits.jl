using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments.db", "hpc/results", true)

nrepeats = 10
config = Dict{Symbol,Any}(
    :nbits => IterableVariable(collect(4:2:16)),
    :nlayers => IterableVariable([4, 8, 12, 16]),
    :repeat_id => IterableVariable(collect(1:nrepeats)),
    :J => 1.0,
    :h => 0.9045,
    :g => 1.4,
    :learning_rate => IterableVariable([0.0002]),
    :use_gpu => true,
    :architecture => IterableVariable([
        [], # No neural network
        [ # 50 neurons
            (; neurons=50, activation=:tanh),
            (; neurons=50, activation=:tanh),
            (; neurons=50, activation=:tanh)
        ],
        [ # 250 neurons
            (; neurons=250, activation=:tanh),
            (; neurons=250, activation=:tanh),
            (; neurons=250, activation=:tanh)
        ],
        [ # 1250 neurons
            (; neurons=1250, activation=:tanh),
            (; neurons=1250, activation=:tanh),
            (; neurons=1250, activation=:tanh)
        ]]),
    :epochs => 200,
)

experiment = Experiment(
    name="NN Optimisation J=h",
    include_file="hpc/nn_optim_trial.jl",
    function_name="run_trial",
    configuration=deepcopy(config)
)

Experimenter.Cluster.init()

@execute experiment db DistributedMode