using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("tuning_experiments.db", "hpc/results", true)

experiment_name = "TFIM NC Learning Rate Tuning 1"

tb_directory = joinpath("hpc", "results", replace(experiment_name, r"\s+"=>"_"))
if !isdir(tb_directory)
    mkdir(tb_directory)
end


nrepeats = 20
config = Dict{Symbol,Any}(
    :nbits => IterableVariable(collect(10)),
    :nlayers => IterableVariable([10, 20, 30, 40]),
    :J => 1.0,
    :h => 0.9045,
    :g => 1.4,
    :epochs => 1000,
    :log_info_freq => 1,
    :tensorboard_directory => tb_directory,
    :weight_init_magnitude => 0.01f0,
    :learning_rate => IterableVariable([0.0002, 0.0001, 0.00005, 0.001, 0.002]),
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
    name=experiment_name,
    include_file="hpc/nn_optim_trial.jl",
    function_name="run_trial",
    configuration=deepcopy(config)
)

Experimenter.Cluster.init()

@execute experiment db MPIMode(1)