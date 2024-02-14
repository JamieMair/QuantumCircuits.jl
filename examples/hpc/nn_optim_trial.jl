using Random
using CUDA
using Flux
using Dates
include("../test_brickwork_problem.jl")
include("../nns/circuit_layer.jl")

function git_sha()
    out = IOBuffer()
    run(pipeline(`git rev-parse HEAD`, stdout=out))
    hash = strip(String(take!(out)))
    return hash
end

function run_trial(config::Dict{Symbol,Any}, trial_id)
    results = Dict{Symbol,Any}()
    nbits = config[:nbits]
    nlayers = config[:nlayers]
    J = config[:J]
    # h = config[:h]
    g = config[:g]
    lr = config[:learning_rate]
    epochs = config[:epochs]

    use_gpu = haskey(config, :use_gpu) ? config[:use_gpu] : CUDA.has_cuda_gpu()
    layer_info = config[:architecture] # list of (; neuron, activation) named tuples


    seed = Int(Random.rand(UInt16))
    results[:seed] = seed
    Random.seed!(seed)

    ngates = QuantumCircuits.brickwork_num_gates(nbits, nlayers)
    nangles = ngates * 15
    results[:ngates] = ngates
    results[:nangles] = nangles

    ψ₀ = QuantumCircuits.zero_state_tensor(nbits)
    if use_gpu
        ψ₀ = CuArray(ψ₀)
    end

    H = TFIMHamiltonian(J, g)
    # H = build_hamiltonian(nbits, J, g)

    initial_layers = []
    last_size = 1
    initial_layers = map(enumerate(layer_info)) do (i, info)
        neurons = info.neurons
        activation = if info.activation == :tanh
            tanh
        elseif info.activation == :σ
            Flux.σ
        elseif info.activation == :relu
            Flux.relu
        else
            error("Unrecognised activation function $(info.activation)")
        end

        last_size = if i == 1
            1
        else
            layer_info[i-1].neurons
        end

        use_bias = haskey(info, :bias) ? info.bias : true

        return Dense(last_size => neurons, activation; bias=use_bias)
    end

    hamiltonian_layer = HamiltonianLayer(nbits, nlayers, ngates, ψ₀, H, QuantumCircuits.construct_grads_cache(ψ₀))

    network = if length(layer_info) == 0
        network = Chain(
            Dense(1 => nangles, identity; bias=false),
            x -> x .* (2π),
            x -> reshape(x, 15, ngates),
            hamiltonian_layer,
            E -> sum(E)
        )
        network
    else
        last_layer_size = layer_info[end].neurons
        network = Chain(
            initial_layers...,
            Dense(last_layer_size => nangles, Flux.σ),
            x -> x .* (2π),
            x -> reshape(x, 15, ngates),
            hamiltonian_layer,
            E -> sum(E))
        network
    end

    network = use_gpu ? (network |> Flux.gpu) : network

    results[:training_start] = now()

    losses = train!(network, epochs; use_gpu, lr=lr)


    results[:training_end] = now()

    results[:duration] = results[:training_end]-results[:training_start]
    results[:duration_s] = round(results[:duration], Dates.Second).value

    results[:energy_trajectory] = losses

    angle_model = Flux.state(network[begin:end-2] |> Flux.cpu)

    results[:model_state] = angle_model
    results[:git_sha] = git_sha()

    # H = network.layers[end-1].H
    # eigen_decomp = eigen(H)
    # min_energy = minimum(eigen_decomp.values)
    # ground_state = eigen_decomp.vectors[:, findfirst(x -> x == min_energy, eigen_decomp.values)]

    # results[:ground_energy] = min_energy
    # results[:ground_state] = ground_state



    return results
end