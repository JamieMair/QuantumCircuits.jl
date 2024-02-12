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
function QuantumCircuits.apply!(ψ′, ψ, circuit::GenericBrickworkCircuit)
    # todo - correct this to use two buffers
    gate_idx = 1
    for l in 1:circuit.nlayers
        for j in circuit_layer_starts(l, circuit.nbits)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), Val(j))
            apply!(ψ′, ψ, gate)
            (ψ′, ψ) = (ψ, ψ′)
            gate_idx += 1
        end
    end
    return ψ
end
function QuantumCircuits.apply(ψ, circuit::GenericBrickworkCircuit)
    ψ′ = similar(ψ)
    ψ′′ = QuantumCircuits.apply!(ψ′, copy(ψ), circuit)
    return ψ′′
end

function reconstruct(circuit::GenericBrickworkCircuit, angle, angle_offset)
    gate_angles = copy(circuit.gate_angles)
    gate_angles[angle] += angle_offset
    GenericBrickworkCircuit(circuit.nbits, circuit.nlayers, circuit.ngates, gate_angles)
end
function measure(H::AbstractMatrix, ψ::AbstractArray)
    @assert ndims(H) == 2
    if ndims(ψ) != 2 || size(ψ, 2) != 1
        ψ = reshape(ψ, :, 1) # reshape to column vector
    end
    measurement = (adjoint(ψ) * (H * ψ))[begin]
    @assert imag(measurement) < 1e-12 # threshold for  imaginary error
    return real(measurement)
end
function measure(H::AbstractMatrix, ψ₀::AbstractArray, circuit::GenericBrickworkCircuit)
    ψ = copy(ψ₀)
    ψ′ = similar(ψ)
    ψ′′ = QuantumCircuits.apply!(ψ′, ψ, circuit);
    
    return measure(H, ψ′′)
end
function gradient(H::AbstractMatrix, ψ₀::AbstractArray, circuit::GenericBrickworkCircuit, gate_index)
    l = reconstruct(circuit, gate_index, π/2);
    r = reconstruct(circuit, gate_index, -π/2);
    (measure(H, ψ₀, l)-measure(H, ψ₀, r)) / 2
end
function gradients(H::AbstractMatrix, ψ₀::AbstractArray, circuit::GenericBrickworkCircuit)
    gs = map(1:length(circuit.gate_angles)) do i
        l = reconstruct(circuit, i, π/2)
        r = reconstruct(circuit, i, -π/2)
        (measure(H, ψ₀, l)-measure(H, ψ₀, r)) / 2
    end

    return reshape(gs, size(circuit.gate_angles))
end
function optimise!(circuit::GenericBrickworkCircuit, H, ψ₀, epochs, lr; use_progress=true)
    energies = Float64[]

    iter = 1:(epochs+1)
    iter = use_progress ? ProgressBar(iter) : iter

    for i in iter 
        energy = measure(H, ψ₀, circuit)
        push!(energies, energy)

        if i <= epochs
            # Gradients
            grad = gradients(H, ψ₀, circuit)

            circuit.gate_angles .-= lr .* grad
        end
    end

    return energies
end