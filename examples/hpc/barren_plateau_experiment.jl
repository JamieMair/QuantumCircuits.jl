using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments_plateaux.db", "hpc/results", true)

nblocks = 20
nrepeats = 5000
nrepeats_per_block = cld(nrepeats, nblocks)

config = Dict{Symbol, Any}(
    :nbits => IterableVariable(collect(4:2:12)),
    :nlayers => IterableVariable([2, 4, 6, collect(8:4:40)...]),
    :J => 1.0,
    :h => 0.9045,
    :g => 1.4,
    :use_gpu => false,
    :weight_init_magnitude => 100.0f0,
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
    :nrepeats => IterableVariable([nrepeats_per_block for _ in 1:nblocks]),
)

experiment = Experiment(
    name="Barren Plateau NNs Full",
    include_file="hpc/barren_plateau_trial.jl",
    function_name="run_trial",
    configuration = deepcopy(config)
)

Experimenter.Cluster.init()

@execute experiment db DistributedMode