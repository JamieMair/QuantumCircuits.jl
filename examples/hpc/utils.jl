using Flux
using QuantumCircuits
include("../nns/circuit_layer.jl")

function git_sha()
    out = IOBuffer()
    run(pipeline(`git rev-parse HEAD`, stdout=out))
    hash = strip(String(take!(out)))
    return hash
end

function create_nn_from_architecture(config)
    nbits = config[:nbits]
    nlayers = config[:nlayers]
    J = config[:J]
    h = config[:h]
    g = config[:g]
    weight_init_magnitude = haskey(config, :weight_init_magnitude) ? config[:weight_init_magnitude] : 0.01f0
    use_gpu = haskey(config, :use_gpu) ? config[:use_gpu] : CUDA.has_cuda_gpu()
    nn_layers_architecture = config[:architecture]
    ψ₀ = QuantumCircuits.zero_state_tensor(nbits)
    if use_gpu
        ψ₀ = CuArray(ψ₀)
    end

    ngates = QuantumCircuits.brickwork_num_gates(nbits, nlayers)
    nangles = ngates * 15

    H = TFIMHamiltonian(J, h, g)


    initial_layers = []
    last_size = 1
    initial_layers = map(enumerate(nn_layers_architecture)) do (i, info)
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
            nn_layers_architecture[i-1].neurons
        end

        use_bias = haskey(info, :bias) ? info.bias : true

        return Dense(last_size => neurons, activation; bias=use_bias, init=Flux.glorot_normal)
    end

    hamiltonian_layer = HamiltonianLayer(nbits, nlayers, ngates, ψ₀, H, QuantumCircuits.construct_grads_cache(ψ₀))

    network = if length(nn_layers_architecture) == 0
        eff_gain = weight_init_magnitude * sqrt((nangles + 1) / 2)
        network = Chain(
            Dense(1 => nangles, identity; bias=false, init=Flux.glorot_normal(gain=eff_gain)),
            x -> x .* (π),
            x -> reshape(x, 15, ngates),
            hamiltonian_layer,
            E -> sum(E)
        )
        network
    else
        last_layer_size = nn_layers_architecture[end].neurons
        eff_gain = weight_init_magnitude * sqrt((nangles + last_layer_size) / 2)
        network = Chain(
            initial_layers...,
            Dense(last_layer_size => nangles, Flux.tanh; init=Flux.glorot_normal(gain=eff_gain)),
            x -> x .* (π),
            x -> reshape(x, 15, ngates),
            hamiltonian_layer,
            E -> sum(E))
        network
    end

    return network
end
