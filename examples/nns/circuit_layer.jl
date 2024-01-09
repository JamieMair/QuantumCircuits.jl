using QuantumCircuits
using ChainRulesCore
using Flux
using ProgressBars
using CUDA

include("../test_brickwork_problem.jl")

struct HamiltonianLayer{T,A<:AbstractArray}
    nbits::Int
    nlayers::Int
    ngates::Int
    ψ₀::A
    H::Matrix{T}
end

function (hl::HamiltonianLayer)(gate_angles::AbstractArray)
    return apply_hamiltonian(hl, gate_angles)
end

function apply_hamiltonian(hl::HamiltonianLayer, gate_angles::AbstractArray)
    @boundscheck size(gate_angles) == (15, hl.ngates)

    circuit = GenericBrickworkCircuit(hl.nbits, hl.nlayers, hl.ngates, gate_angles)
    
    E = QuantumCircuits.measure(hl.H, hl.ψ₀, circuit);
    return E
end
function apply_hamiltonian(hl::HamiltonianLayer, gate_angles::CuArray)
    return apply_hamiltonian(hl, Array(gate_angles))
end

function ChainRulesCore.rrule(::typeof(apply_hamiltonian), hl::HamiltonianLayer, gate_angles::CuArray)
    angles_cpu = Array(gate_angles)
    E = apply_hamiltonian(hl, angles_cpu)
    function pb(dE)
        gs = if dE == 1
            ChainRulesCore.@thunk(Flux.gpu(QuantumCircuits.gradients(hl.H, hl.ψ₀, GenericBrickworkCircuit(hl.nbits, hl.nlayers, hl.ngates, angles_cpu))))
        else
            ChainRulesCore.@thunk(Flux.gpu(Array(dE) .* QuantumCircuits.gradients(hl.H, hl.ψ₀, GenericBrickworkCircuit(hl.nbits, hl.nlayers, hl.ngates, angles_cpu))))
        end

        return NoTangent(), NoTangent(), gs
    end
    return E, pb
end

function ChainRulesCore.rrule(::typeof(apply_hamiltonian), hl::HamiltonianLayer, gate_angles::AbstractArray)
    E = apply_hamiltonian(hl, gate_angles)
    function pb(dE)
        gs = if dE == 1
            ChainRulesCore.@thunk(QuantumCircuits.gradients(hl.H, hl.ψ₀, GenericBrickworkCircuit(hl.nbits, hl.nlayers, hl.ngates, gate_angles)))
        else
            ChainRulesCore.@thunk(dE .* QuantumCircuits.gradients(hl.H, hl.ψ₀, GenericBrickworkCircuit(hl.nbits, hl.nlayers, hl.ngates, gate_angles)))
        end

        return NoTangent(), NoTangent(), gs
    end
    return E, pb
end

Flux.@functor HamiltonianLayer

function HamiltonianLayer(nbits::Int, nlayers::Int, ngates::Int, ψ₀::CuArray, H::CuArray)
    return HamiltonianLayer(nbits, nlayers, ngates, Array(ψ₀), Array(H))
end
function HamiltonianLayer(nbits::Int, nlayers::Int, ngates::Int, ψ₀::Array, H::CuArray)
    return HamiltonianLayer(nbits, nlayers, ngates, ψ₀, Array(H))
end
function HamiltonianLayer(nbits::Int, nlayers::Int, ngates::Int, ψ₀::CuArray, H::Array)
    return HamiltonianLayer(nbits, nlayers, ngates, Array(ψ₀), H)
end

function test_network(;use_gpu = true)
    nbits = 10
    nlayers = 10
    ngates = QuantumCircuits.brickwork_num_gates(nbits, nlayers)
    nangles = ngates * 15
    ψ₀ = QuantumCircuits.zero_state_tensor(nbits)
    J = 1;
    h = 0.5;
    g = 0;
    H = build_hamiltonian(nbits, J, h, g);
    model = Chain(
        Dense(1=>50, tanh),
        Dense(50=>50, tanh),
        Dense(50=>nangles, Flux.σ),
        x -> x .* (2π),
        x -> reshape(x, 15, ngates),
        HamiltonianLayer(nbits, nlayers, ngates, ψ₀, H),
        E -> sum(E)
    )
    use_gpu && return model |> Flux.gpu
    return model
end

function train!(network, epochs; lr=0.005, use_gpu = true)
    input = use_gpu ? [1.0f0] |> Flux.gpu : [1.0f0];
    
    losses = Float32[];
    optim = Flux.setup(Flux.Adam(lr), network);
    for e in ProgressBar(1:epochs)
        energy, grads = Flux.withgradient(network) do m
            m(input)
        end
        Flux.update!(optim, network, grads[1])
        push!(losses, energy)
    end
    return losses
end


network = test_network();
epochs = 100
losses = train!(network, epochs);

H = network.layers[end-1].H
eigen_decomp = eigen(H);
min_energy = minimum(eigen_decomp.values);
ground_state = eigen_decomp.vectors[:, findfirst(x->x==min_energy, eigen_decomp.values)]

using CairoMakie
using LaTeXStrings
begin
    f = Figure()
    ax = Axis(f[1,1],
        title="Gradient Descent with a neural network",
        xlabel="# Epochs",
        ylabel=L"\left \langle H \right \rangle")

    
    lines!(ax, 0:(length(losses)-1), losses, label=L"\langle H \ \rangle", color=:black)
    hlines!(ax, [min_energy], label=L"E_0", linestyle=:dash)
    xlims!(ax, (0, epochs))

    f
end