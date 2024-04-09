using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments_noisy.db", "hpc/results", true)

nrepeats = 10
config = Dict{Symbol,Any}(
    :nbits => IterableVariable(collect(4:2:8)),
    :nlayers => IterableVariable([4, 12, 20]),
    :J => 1.0,
    :h => 0.9045,
    :g => 1.4,
    :epochs => 400,
    :save_grads_freq => 1,
    :weight_init_magnitude => 0.01f0,
    :hamiltonian_noise => IterableVariable([0, 0.1, 0.5]),
    :learning_rate => IterableVariable([0.0002]),
    :use_gpu => false,
    :architecture => IterableVariable([
        [], # No neural network
        [ # 50 neurons
            (; neurons=50, activation=:tanh),
            (; neurons=50, activation=:tanh),
            (; neurons=50, activation=:tanh)
        ],
    ]),
    :repeat_id => IterableVariable(collect(1:nrepeats)),
)

experiment = Experiment(
    name="NN Noisy Optim Longer Test",
    include_file="hpc/nn_optim_trial.jl",
    function_name="run_trial",
    configuration=deepcopy(config)
)

# Experimenter.Cluster.init()

@execute experiment db DistributedMode