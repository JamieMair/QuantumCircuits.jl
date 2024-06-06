using Random
using CUDA
using Flux
using Dates
using LinearAlgebra
using SparseArrays
using QuantumCircuits
include("../matrix_tfim.jl")
include("../nns/circuit_layer.jl")
include("../nns/training.jl")
include("utils.jl")

function run_trial(config::Dict{Symbol,Any}, trial_id)
    results = Dict{Symbol,Any}()
    nbits = config[:nbits]
    nlayers = config[:nlayers]
    lr = config[:learning_rate]
    epochs = config[:epochs]
    log_info_freq = config[:log_info_freq]

    use_gpu = haskey(config, :use_gpu) ? config[:use_gpu] : CUDA.has_cuda_gpu()

    seed = Int(Random.rand(UInt16))
    results[:seed] = seed
    Random.seed!(seed)

    ngates = QuantumCircuits.brickwork_num_gates(nbits, nlayers)
    nangles = ngates * 15
    results[:ngates] = ngates
    results[:nangles] = nangles

    
    network = create_nn_from_architecture(config)

    logger = if haskey(config, :tensorboard_directory)
        if !isdir(config[:tensorboard_directory])
            error("[ERROR] Could not find the log directory at $(abspath(config[:tensorboard_directory]))")
        end
        custom_log_directory = joinpath(config[:tensorboard_directory], string(trial_id))
        results[:logger_directory] = custom_log_directory

        logger = TrainingTBLogger(TBLogger(custom_log_directory, tb_append))
        logger
    else
        NullLogger()
    end

    # Extract hparams
    hyperparameters = extract_hyperparameters(network, config)
    log_hyperparameters!(logger, hyperparameters, ["metrics/energy"])

    results[:training_start] = now()

    losses, info = train!(network, epochs, logger; use_gpu, lr, log_info_freq, use_progress=false)

    results[:training_end] = now()

    results[:duration] = results[:training_end]-results[:training_start]
    results[:duration_s] = round(results[:duration], Dates.Second).value

    results[:energy_trajectory] = losses
    results[:training_info] = info

    # Get the important part of the architecture
    angle_model = Flux.state(network[begin:end-2] |> Flux.cpu)

    # Calculate number of parameters in the neural network
    parameters, _ = Flux.destructure(angle_model)
    nparams = length(parameters)

    results[:nparams] = nparams
    results[:model_state] = angle_model
    results[:git_sha] = git_sha()

    return results
end