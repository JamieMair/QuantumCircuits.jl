function optimise!(circuit::GenericBrickworkCircuit, H, ψ₀, epochs, lr; use_progress=true)
    iter = 1:(epochs)
    iter = use_progress ? ProgressBar(iter) : iter
    
    energies = zeros(Float64, length(iter)+1)

    cache_args = construct_grads_cache(ψ₀, circuit)

    for i in iter 
        E, grads = gradients!(cache_args..., H, circuit)
        energies[i+1] = E

        circuit.gate_angles .-= lr .* grads
    end
    energies[end] = measure(H, ψ₀, circuit)
    return energies
end

function build_identity(dtype::DataType, nbits)
    I = zeros(dtype, 2^nbits, 2^nbits)
    for i in 1:2^nbits
        I[i, i] = 1
    end
    return I
end

function LinearAlgebra.adjoint(gate::Generic2SpinGate)
    return Generic2SpinGate(reshape(adjoint(reshape(gate.array, 4, 4)), 2, 2, 2, 2))
end
function LinearAlgebra.adjoint(gate::Localised2SpinAdjGate)
    return Localised2SpinAdjGate(adjoint(gate.gate), gate.target_gate_dim)
end

function combine_gates(gate_a::Generic2SpinGate, gate_b::Generic2SpinGate)
    c = similar(gate_a.array, promote_type(eltype(gate_a.array), eltype(gate_b.array)))
    c .= zero(eltype(c))

    a = gate_a.array
    b = gate_b.array

    @inbounds for (c1, c2, c3, c4) in product((1:2 for _ in 1:4)...)
        x = a[c1, c2, c3, c4]
        for j in 1:2
            for i in 1:2
                # u is reversed as Julia is column-major unlike row major of numpy
                c[c1, i, j, c2] += x * b[i, j, c1, c2]
            end
        end
    end
    return Generic2SpinGate(c)
end


function propagate_forwards!(cache, ψ′, ψ, circuit::GenericBrickworkCircuit, start_layer::Int, layer_gate_idx::Int, gate_idx::Int)
    for l in start_layer:circuit.nlayers
        # Get an iterator over starting gate positions in this layer, skipping applied gates
        iter = if l == start_layer
            @views circuit_layer_starts(l, circuit.nbits)[layer_gate_idx:end]
        else
            circuit_layer_starts(l, circuit.nbits)
        end
        for j in iter
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), j)
            apply!(cache, ψ′, ψ, gate)
            (ψ′, ψ) = (ψ, ψ′)
            gate_idx += 1
        end
    end
    return ψ, ψ′
end

function construct_grads_cache(ψ, circuit)
    gradients = similar(circuit.gate_angles)

    ψ = copy(ψ) # Don't mutate initial state
    ψ′ = similar(ψ) # Create a buffer for storing intermediate results
    ψ′′ = similar(ψ) # Create a restore point for the state

    cache = construct_apply_cache(ψ)

    return (cache, gradients, ψ′, ψ′′, ψ)
end

function gradients(H, ψ, circuit::GenericBrickworkCircuit)
    cache_args = construct_grads_cache(ψ, circuit)
    return gradients!(cache_args..., H, circuit)
end

function gradients!(cache, gradients, ψ′, ψ′′, ψ, H, circuit::GenericBrickworkCircuit)
    # Complete a pass through the circuit
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

    # Measure the energy
    E = measure!(ψ′, H, ψ)
    
    gate_idx = size(circuit.gate_angles, 2) # set to last gate
    for l in circuit.nlayers:-1:1
        for (layer_gate_num, j) in reverse(collect(enumerate(circuit_layer_starts(l, circuit.nbits))))
            angles = view(circuit.gate_angles, :, gate_idx)
            original_gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), j)
            gate_dagger = adjoint(original_gate)

            # Undo the current gate by applying the Hermitian conjugate
            apply!(cache, ψ′, ψ, gate_dagger)
            (ψ′, ψ) = (ψ, ψ′)

            # Set a restore point for ψ
            ψ′′ .= ψ

            for k in axes(angles, 1)
                original_angle = angles[k]

                # Calculate first energy
                angles[k] = original_angle + π/2
                ψ, ψ′ = propagate_forwards!(cache, ψ′, ψ, circuit, l, layer_gate_num, gate_idx)
                E_plus = measure!(ψ′, H, ψ)

                # Reset to original state
                ψ .= ψ′′
                angles[k] = original_angle - π/2
                ψ, ψ′ = propagate_forwards!(cache, ψ′, ψ, circuit, l, layer_gate_num, gate_idx)
                E_minus = measure!(ψ′, H, ψ)
                
                # Calculate gradient using formula
                gradients[k, gate_idx] = (E_plus - E_minus) / 2

                # Reset for next angle
                ψ .= ψ′′
                angles[k] = original_angle
            end

            # Go to the previous gate
            gate_idx -= 1
        end
    end

    return E, gradients
end