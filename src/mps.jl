# add package using "add https://github.com/AdamSmith-physics/MatrixProductStates.jl#main"
using MatrixProductStates
using LinearAlgebra

export measure
function measure(H::Hamiltonian, psi::MPS)
    n = length(psi)
    n == H.nbits || throw(ArgumentError("State and Hamiltonian have different numbers of qubits"))

    sort!(H.termList, by=x->x.index)  # sort terms by index

    # loop over terms in Hamiltonian
    energy = 0.0
    #MatrixProductStates.movecentre!(psi, 1)  # not necessary!
    for term in H.termList
        if size(term.matrix) == (2,2)
            # single site term
            energy += expectation_1site(psi, term.matrix, term.index)
        elseif size(term.matrix) == (4,4)
            # two site term
            energy += expectation_2site(psi, term.matrix, term.index)
        else
            throw(ArgumentError("Invalid matrix size for term"))
        end
    end
    return real(energy)  # assumers H is Hermitian!
end

export apply!
function apply!(psi::MPS, circuit::GenericBrickworkCircuit; normalised::Bool=false)
    
    gate_idx = 1
    for l in 1:circuit.nlayers
        for j in (1 + (l-1) % 2):(circuit.nbits-1)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = mat(build_general_unitary_gate(angles))
            
            gate = permutedims(gate, (2,1,4,3))
            gate = reshape(gate, (4,4))

            apply_2site!(psi, gate, j; normalised=normalised)
            gate_idx += 1
        end
    end
end

"""export build_general_unitary
function build_general_unitary(angles::AbstractVector)

    # 12 angles used
    rotation_gates = map(0:3) do i
        offset = 3*i
        rotation_gate(angles[offset+1],angles[offset+2], angles[offset+3])
    end
    # Remaining 3 angles used
    theta = angles[13]
    phi = angles[14]
    lambda = angles[15]

    l1 = kron(RZGate(T(-π/2)) * rotation_gates[4], rotation_gates[3])
    l2 = kron(RYGate(phi), RZGate(theta))
    l3 = kron(RYGate(lambda), IdentityGate())
    l4 = kron(rotation_gates[2], rotation_gates[1]*RZGate(T(-π/2)))

end


function rotation_gate(theta, phi, lambda)
    cos_theta_term = cos(theta / 2)
    sin_theta_term = sin(theta / 2)
    phase_1 = exp(im*lambda)
    phase_2 = exp(im*phi)
    phase_3 = phase_1 * phase_2
    return [cos_theta_term -phase_1*sin_theta_term; phase_2*sin_theta_term cos_theta_term * phase_3]
end
"""