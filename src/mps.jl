# add package using "add https://github.com/AdamSmith-physics/MatrixProductStates.jl#main"
using MatrixProductStates
using LinearAlgebra


#should be added to MatrixProductStates.jl
function Base.copy(psi::MPS)
    return MPS(psi.d, psi.N, copy(psi.tensors), psi.centre, psi.chiMax, psi.threshold)
end


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

function measure(H::Hamiltonian, psi::MPS, circuit::GenericBrickworkCircuit)
    psi_copy = copy(psi)
    apply!(psi_copy, circuit)
    return measure(H, psi_copy)
end

function measure(H::Hamiltonian, circuit::GenericBrickworkCircuit; chiMax::Int=0, threshold::Real=0.0)
    psi = MPS(H.nbits)
    psi.chiMax = chiMax
    psi.threshold = threshold
    return measure(H, psi, circuit)
end


export apply!
function apply!(psi::MPS, circuit::GenericBrickworkCircuit; normalised::Bool=false)
    
    gate_idx = 1
    for l in 1:circuit.nlayers
        """for j in (1 + (l-1) % 2):2:(circuit.nbits-1)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = mat(build_general_unitary_gate(angles))
            
            gate = permutedims(gate, (2,1,4,3))
            gate = reshape(gate, (4,4))

            apply_2site!(psi, gate, j; normalised=normalised)
            gate_idx += 1
        end"""
        layer_gates = []
        for j in (1 + (l-1) % 2):2:(circuit.nbits-1)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = mat(build_general_unitary_gate(angles))
            gate = permutedims(gate, (2,1,4,3))
            gate = reshape(gate, (4,4))
            new_term = Term(j,gate)
            push!(layer_gates, new_term)
            gate_idx += 1
        end
        if psi.centre <= psi.N/2
            sort!(layer_gates, by=x->x.index)
        else
            sort!(layer_gates, by=x->-x.index)
        end
        for term in layer_gates
            #println("overlap = $(overlap(psi,psi))")
            apply_2site!(psi, term.matrix, term.index; normalised=normalised)
            #println("overlap = $(overlap(psi,psi)), site = $(term.index), center = $(psi.centre)")
        end

    end
end

#export gradient
function gradient(H::Hamiltonian, psi::MPS, circuit::GenericBrickworkCircuit, gate_index)
    l = reconstruct(circuit, gate_index, π/2)
    r = reconstruct(circuit, gate_index, -π/2)
    (measure(H, psi, l)-measure(H, psi, r)) / 2
end

function gradient(H::Hamiltonian, circuit::GenericBrickworkCircuit, gate_index; chiMax::Int=0, threshold::Real=0.0)
    psi = MPS(H.nbits)
    psi.chiMax = chiMax
    psi.threshold = threshold
    return gradient(H, psi, circuit, gate_index)
end


#export gradients
function gradients(H::Hamiltonian, psi::MPS, circuit::GenericBrickworkCircuit; calculate_energy::Bool = false)
    gs = map(1:length(circuit.gate_angles)) do i
        l = reconstruct(circuit, i, π/2)
        r = reconstruct(circuit, i, -π/2)
        (measure(H, psi, l)-measure(H, psi, r)) / 2
    end

    grads = reshape(gs, size(circuit.gate_angles))
    if calculate_energy
        E = measure(H, psi, circuit)
        return E, grads
    else
        return grads
    end
end

function gradients(H::Hamiltonian, circuit::GenericBrickworkCircuit; chiMax::Int=0, threshold::Real=0.0, calculate_energy::Bool = false)
    psi = MPS(H.nbits)
    psi.chiMax = chiMax
    psi.threshold = threshold
    grads = gradients(H, psi, circuit)
    if calculate_energy
        E = measure(H, psi, circuit)
        return E, grads
    else
        return grads
    end
end

