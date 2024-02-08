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
function LinearAlgebra.adjoint(gate::Localised2SpinAdjGate{G, K}) where {G, K}
    return Localised2SpinAdjGate(adjoint(gate.gate), Val(K))
end


function calculate_grads!(ψ, circuit::GenericBrickworkCircuit, H::AbstractMatrix)
    gradients = similar(circuit.gate_angles)
    # todo - correct this to use two buffers
    ψ′ = similar(ψ)

    # Complete a pass through the circuit
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

    # Measure the energy
    E = measure(H, ψ)

    # Calculate the hermitian conjugate gates
    M, M′ = copy(H), similar(H)
    is_final_layer = true

    
    gate_idx = size(circuit.gate_angles, 2) # set to last gate
    for l in circuit.nlayers:-1:1
        for j in reverse(circuit_layer_starts(l, circuit.nbits))
            angles = view(circuit.gate_angles, :, gate_idx)
            original_gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), Val(j))
            gate_dagger = adjoint(original_gate)

            # Undo the current gate by applying the Hermitian conjugate
            apply!(ψ′, ψ, gate_dagger)
            (ψ′, ψ) = (ψ, ψ′)

            for k in axes(angles, 1)
                original_angle = angles[k, gate_idx]

                # Calculate first energy
                angles[k, gate_idx] = original_angle + π/2
                gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), Val(j))
                apply!(ψ′, ψ, gate)
                (ψ′, ψ) = (ψ, ψ′)
                
                E_plus = dot(reshape(ψ, :, 1), M, reshape(ψ, :, 1))

                # Undo the plus gate and apply the second gate
                angles[k, gate_idx] = original_angle - π/2
                minus_gate = build_general_unitary_gate(angles)
                gate = Localised2SpinAdjGate(Generic2SpinGate(reshape(reshape(minus_gate.array, 4, 4) * adjoint(reshape(gate.array, 4, 4)), 2, 2, 2, 2)), Val(j))
                apply!(ψ′, ψ, gate)
                (ψ′, ψ) = (ψ, ψ′)

                E_minus = dot(reshape(ψ, :, 1), M, reshape(ψ, :, 1))

                # Undo the final gate
                gate = Localised2SpinAdjGate(Generic2SpinGate(reshape(adjoint(reshape(minus_gate.array, 4, 4)), 2, 2, 2, 2)), Val(j))
                apply!(ψ′, ψ, gate)
                (ψ′, ψ) = (ψ, ψ′)

                angles[k, gate_idx] = original_angle
                gradients[k, gate_idx] = (E_plus - E_minus) / 2
            end

            # Apply the current gate to the matrix
            right_apply_gate!(M′, M, original_gate)

            # Go to the second last gate
            gate_idx -= 1
        end

        is_final_layer = false
    end

    return E, gradients
end