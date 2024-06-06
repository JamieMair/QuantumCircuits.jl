using QuantumCircuits
using ChainRulesCore
using Flux
using ProgressBars
using CUDA
using Statistics

struct HamiltonianLayer{CT,TH,A<:AbstractArray}
    nbits::Int
    nlayers::Int
    ngates::Int
    ψ₀::A
    H::TH
    cache::CT
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
    E, gradients = QuantumCircuits.gradients!(hl.cache..., hl.H, hl.ψ₀, GenericBrickworkCircuit(hl.nbits, hl.nlayers, hl.ngates, Array(gate_angles)); calculate_energy=true)
    function pb(dE)
        gs = if dE == 1
            ChainRulesCore.@thunk(Flux.gpu(gradients))
        else
            ChainRulesCore.@thunk(dE .* Flux.gpu(gradients))
        end

        return NoTangent(), NoTangent(), gs
    end
    return E, pb
end

function ChainRulesCore.rrule(::typeof(apply_hamiltonian), hl::HamiltonianLayer, gate_angles::AbstractArray)
    E, gradients = QuantumCircuits.gradients!(hl.cache..., hl.H, hl.ψ₀, GenericBrickworkCircuit(hl.nbits, hl.nlayers, hl.ngates, gate_angles); calculate_energy=true)
    function pb(dE)
        gs = if dE == 1
            ChainRulesCore.@thunk(gradients)
        else
            ChainRulesCore.@thunk(dE .* gradients)
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