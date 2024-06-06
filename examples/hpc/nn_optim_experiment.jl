using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments_long.db", "hpc/results", true)

nrepeats = 100
config = Dict{Symbol,Any}(
    :nbits => IterableVariable(collect(4:2:12)),
    :nlayers => IterableVariable([4, 8, 12, 16, 20]),
    :J => 1.0,
    :h => 0.9045,
    :g => 1.4,
    :epochs => 200,
    :log_info_freq => 1,
    :weight_init_magnitude => 0.01f0,
    :learning_rate => IterableVariable([0.0002]),
    :use_gpu => false,
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
        ]
    ]),
    :repeat_id => IterableVariable(collect(1:nrepeats)),
)

experiment = Experiment(
    name="NN Optimisation Near Critical With Gradients",
    include_file="hpc/nn_optim_trial.jl",
    function_name="run_trial",
    configuration=deepcopy(config)
)

Experimenter.Cluster.init()

@execute experiment db DistributedMode