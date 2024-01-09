using Distributed
using ClusterManagers


num_tasks = parse(Int, ENV["SLURM_NTASKS"]) # One process per task
cpus_per_task = parse(Int, ENV["SLURM_CPUS_PER_TASK"]) # Assign threads per process
addprocs(ClusterManagers.SlurmManager(num_tasks),
    exeflags=[
    "--project",
    "--threads=$cpus_per_task"]
    )



using Experimenter
cd(joinpath(@__DIR__, ".."))

db = open_db("experiments.db", "hpc/results", true)

nrepeats = 10
config = Dict{Symbol, Any}(
    :nbits => IterableVariable(collect(4:2:12)),
    :nlayers => IterableVariable([4, 8, 12, 16, 20, 24, 28, 32, 36, 40]),
    :repeat_id => IterableVariable(collect(1:nrepeats)),
    :J => 1.0,
    :h => 0.5,
    :g => 0.0,
    :use_gpu => true,
    :architecture => [
        (; neurons = 50, activation = :tanh),
        (; neurons = 50, activation = :tanh),
        (; neurons = 50, activation = :tanh)
    ],
    :epochs = 80,
)

experiment = Experiment(
    name="Barren Plateau Vanilla",
    include_file="hpc/barren_plateau_trial.jl",
    function_name="run_trial",
    configuration = deepcopy(config)
)

@execute experiment db DistributedMode