function build_identity(dtype::DataType, nbits)
    I = zeros(dtype, 2^nbits, 2^nbits)
    for i in 1:2^nbits
        I[i, i] = 1
    end
    return I
end

function LinearAlgebra.adjoint(gate::Generic2SpinGate)
    # adj = permutedims(conj.(gate.array), (4, 3, 2, 1))
    return Generic2SpinGate(reshape(adjoint(reshape(gate.array, 4, 4)), 2, 2, 2, 2))
    # return Generic2SpinGate(adj)
end
function LinearAlgebra.adjoint(gate::Localised2SpinAdjGate{G, K}) where {G, K}
    return Localised2SpinAdjGate(adjoint(gate.gate), Val(K))
end

function combine_gates(gate_a::Generic2SpinGate, gate_b::Generic2SpinGate)
    c = similar(gate_a.array, promote_type(eltype(gate_a.array), eltype(gate_b.array)))
    c .= zero(eltype(c))

    a = gate_a.array
    b = gate_b.array

    for (c1, c2, c3, c4) in product((1:2 for _ in 1:4)...)
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

measure_energy(H::AbstractMatrix, ψ) = measure_energy(H, reshape(ψ, :))
function measure_energy(H::AbstractMatrix, ψ::AbstractVector)
    E = dot(conj(ψ), H, ψ)
    @assert imag(E) < 1e-15 "The energy must be approximately real"
    return real(E)
end
measure_energy!(ψ′, H::AbstractMatrix, ψ) = measure_energy!(reshape(ψ′, :), H, reshape(ψ, :))
function measure_energy!(ψ′::AbstractVector, H::AbstractMatrix, ψ::AbstractVector)
    mul!(ψ′, H, ψ)
    E = sum((x) -> conj(x[1]) * x[2], zip(ψ′, ψ))
    @assert imag(E) < 1e-10 "The energy must be approximately real"
    return real(E)
end

function calculate_grads(H::AbstractMatrix, ψ, circuit::GenericBrickworkCircuit)
    gradients = similar(circuit.gate_angles)
    # todo - correct this to use two buffers
    ψ = copy(ψ) # don't mutate initial state
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
    E = measure_energy!(ψ′, H, ψ)

    # Calculate the hermitian conjugate gates
    M, M′ = similar(H, ComplexF64), similar(H, ComplexF64)
    M .= H # copy over the H values

    
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
                original_angle = angles[k]

                # Calculate first energy
                angles[k] = original_angle + π/2
                gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), Val(j))
                apply!(ψ′, ψ, gate)
                (ψ′, ψ) = (ψ, ψ′)
                
                E_plus = measure_energy!(ψ′, M, ψ)

                # Undo the plus gate and apply the second gate
                apply!(ψ′, ψ, adjoint(gate))
                (ψ′, ψ) = (ψ, ψ′)
                

                angles[k] = original_angle - π/2
                gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), Val(j))
                # gate = Localised2SpinAdjGate(combine_gates(adjoint(gate.gate), minus_gate), Val(j))
                # gate = Localised2SpinAdjGate(Generic2SpinGate(reshape(reshape((adjoint(gate)).gate.array, 4, 4) * reshape(minus_gate.array, 4, 4), 2, 2, 2, 2)), Val(j))
                apply!(ψ′, ψ, gate)
                (ψ′, ψ) = (ψ, ψ′)

                E_minus = measure_energy!(ψ′, M, ψ)

                # Undo the final gate
                apply!(ψ′, ψ, adjoint(gate))
                (ψ′, ψ) = (ψ, ψ′)

                angles[k] = original_angle
                gradients[k, gate_idx] = real((E_plus - E_minus) / 2)
            end

            # Apply the current gate to the matrix
            right_apply_gate!(M′, M, original_gate)
            (M′, M) = (M, M′) # Swap the arrays
            
            # Go to the second last gate
            gate_idx -= 1
        end
    end

    return E, gradients
end