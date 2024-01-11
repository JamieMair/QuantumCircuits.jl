using QuantumCircuits
using ChainRulesCore
using Flux
using ProgressBars
using CUDA

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

function train!(network, epochs; lr=0.01, use_gpu = true, use_progress=false)
    input = use_gpu ? [1.0f0] |> Flux.gpu : [1.0f0];
    
    losses = Float32[];
    optim = Flux.setup(Flux.Adam(lr), network);
    iter = use_progress ? ProgressBar(1:epochs) : (1:epochs)
    for e in iter
        energy, grads = Flux.withgradient(network) do m
            m(input)
        end
        Flux.update!(optim, network, grads[1])
        push!(losses, energy)
    end
    return losses
end



# using CairoMakie
# using LaTeXStrings
# begin
#     f = Figure()
#     ax = Axis(f[1,1],
#         title="Gradient Descent with a neural network",
#         xlabel="# Epochs",
#         ylabel=L"\left \langle H \right \rangle")

    
#     lines!(ax, 0:(length(losses)-1), losses, label=L"\langle H \ \rangle", color=:black)
#     hlines!(ax, [min_energy], label=L"E_0", linestyle=:dash)
#     xlims!(ax, (0, epochs))

#     f
# end