using Random
using CUDA
using QuantumCircuits
import LinearAlgebra: norm
import LinearAlgebra
include("utils.jl")

LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

function gate_gradient(nbits, nlayers, H, ψ₀, index=1)
    circuit = GenericBrickworkCircuit(nbits, nlayers)
    Random.rand!(circuit.gate_angles)
    circuit.gate_angles .*= 2 * π
    return gradient(H, ψ₀, circuit, index)
end

function git_sha()
    out=IOBuffer()
    run(pipeline(`git rev-parse HEAD`, stdout=out))
    hash = strip(String(take!(out)))
    return hash
end

function run_trial(config::Dict{Symbol, Any}, trial_id) 
    results = Dict{Symbol, Any}()
    nbits = config[:nbits]
    nlayers = config[:nlayers]
    nrepeats = config[:nrepeats]
    J = config[:J]
    h = config[:h]
    g = config[:g]
    use_gpu = haskey(config, :use_gpu) ? config[:use_gpu] : false


    seed = Int(Random.rand(UInt16))
    results[:seed] = seed
    Random.seed!(seed)

    results[:git_sha] = git_sha()

    ngates = QuantumCircuits.brickwork_num_gates(nbits, nlayers)
    nangles = ngates * 15
    results[:ngates] = ngates
    results[:nangles] = nangles

    
    input = use_gpu ? [1.0f0;;] |> Flux.gpu : [1.0f0;;];
    energies = zeros(Float32, nrepeats)
    gradients = zeros(Float32, nrepeats)
    number_of_params = 0

    for i in 1:nrepeats
        network = create_nn_from_architecture(config)
        energy, grads = Flux.withgradient(network) do m 
            m(input)
        end
        gs, _ = Flux.destructure(grads)
        norm_gradient = norm(reshape(gs, :)) / length(gs)
        number_of_params = length(gs)

        energies[i] = energy
        gradients[i] = norm_gradient
    end

    results[:number_of_params] = number_of_params
    results[:gradients] = gradients
    results[:energies] = energies

    return results
end