using Random
using CUDA
using Flux
using Dates
using LinearAlgebra
using SparseArrays
using QuantumCircuits
include("../matrix_tfim.jl")
include("../nns/circuit_layer.jl")
include("utils.jl")

function run_trial(config::Dict{Symbol,Any}, trial_id)
    results = Dict{Symbol,Any}()
    nbits = config[:nbits]
    nlayers = config[:nlayers]
    J = config[:J]
    h = config[:h]
    g = config[:g]
    lr = config[:learning_rate]
    epochs = config[:epochs]
    save_grads_freq = config[:save_grads_freq]

    use_gpu = haskey(config, :use_gpu) ? config[:use_gpu] : CUDA.has_cuda_gpu()

    seed = Int(Random.rand(UInt16))
    results[:seed] = seed
    Random.seed!(seed)

    ngates = QuantumCircuits.brickwork_num_gates(nbits, nlayers)
    nangles = ngates * 15
    results[:ngates] = ngates
    results[:nangles] = nangles

    
    network = create_nn_from_architecture(config)

    results[:training_start] = now()

    losses, info = train!(network, epochs; use_gpu, lr, save_grads_freq, use_progress=false)


    results[:training_end] = now()

    results[:duration] = results[:training_end]-results[:training_start]
    results[:duration_s] = round(results[:duration], Dates.Second).value

    results[:energy_trajectory] = losses
    results[:training_info] = info

    angle_model = Flux.state(network[begin:end-2] |> Flux.cpu)

    results[:model_state] = angle_model
    results[:git_sha] = git_sha()

    if nbits <= 10
        H = build_hamiltonian(nbits, J, h, g)
        eigen_decomp = eigen(H)
        min_energy = minimum(eigen_decomp.values)
        ground_state = eigen_decomp.vectors[:, findfirst(x -> x == min_energy, eigen_decomp.values)]

        results[:ground_energy] = min_energy
        results[:ground_state] = ground_state
    end



    return results
end