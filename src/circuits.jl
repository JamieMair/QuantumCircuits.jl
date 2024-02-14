using ProgressBars

struct GenericBrickworkCircuit{T<:Real}
    nbits::Int
    nlayers::Int
    ngates::Int
    gate_angles::Matrix{T}
end
circuit_layer_starts(layer_number, nbits) = (1 + (layer_number-1) % 2):2:(nbits-1)
function brickwork_num_gates(nbits, nlayers)
    return sum(l->length(circuit_layer_starts(l, nbits)), 1:nlayers)
end
function GenericBrickworkCircuit(nbits, nlayers)
    ngates = brickwork_num_gates(nbits, nlayers)

    gate_array = zeros(Float64, 15, ngates);
    return GenericBrickworkCircuit(nbits, nlayers, ngates, gate_array)
end
function QuantumCircuits.apply!(cache::CPUApplyCache, ψ′::AbstractArray, ψ::AbstractArray, circuit::GenericBrickworkCircuit)    
    gate_idx = 1
    for l in 1:circuit.nlayers
        for j in circuit_layer_starts(l, circuit.nbits)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), j)
            apply!(cache, ψ′, ψ, gate)
            (ψ′, ψ) = (ψ, ψ′)
            gate_idx += 1
        end
    end
    return ψ
end
"""
    Applies the given circuit to the state ψ.

The original state is not modified, unlike in the `apply!` version.
"""
function QuantumCircuits.apply(ψ::AbstractArray, circuit::GenericBrickworkCircuit)
    cache = construct_apply_cache(ψ)
    ψ = similar(ψ)
    ψ′ = similar(ψ)
    return QuantumCircuits.apply!(cache, ψ′, ψ, circuit)
end

function reconstruct(circuit::GenericBrickworkCircuit, angle, angle_offset)
    gate_angles = copy(circuit.gate_angles)
    gate_angles[angle] += angle_offset
    GenericBrickworkCircuit(circuit.nbits, circuit.nlayers, circuit.ngates, gate_angles)
end
function measure(H::AbstractMatrix, ψ::AbstractArray)
    @assert ndims(H) == 2
    ψ = reshape(ψ, :)
    measurement = dot(ψ, H, ψ)
    @assert imag(measurement) < 1e-12 # threshold for  imaginary error
    return real(measurement)
end
function measure!(ψ′::AbstractArray, H::AbstractMatrix, ψ::AbstractArray)
    return measure(H, ψ)
end
function measure(H::AbstractMatrix, ψ::AbstractArray, circuit::GenericBrickworkCircuit)
    ψ′ = QuantumCircuits.apply(ψ, circuit)
    return measure(H, ψ′)
end


