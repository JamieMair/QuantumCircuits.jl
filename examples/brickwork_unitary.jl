using Revise
using QuantumCircuits
using Random
using LinearAlgebra


function build_hamiltonian(n, J, h, g=0)
    H = zeros(2^n, 2^n)
    for i in 1:n-1
        a = Localised1SpinGate(XGate(), Val(i))
        b = Localised1SpinGate(XGate(), Val(i+1))
        H .+= convert_gates_to_matrix(n, Localised1SpinGate[a,b])
    end
    H .*= -J

    for i in 1:n
        a = Localised1SpinGate(ZGate(), Val(i))
        H .+= h .* convert_gates_to_matrix(n, Localised1SpinGate[a])
    end
    
    if g != 0
        for i in 1:n
            a = Localised1SpinGate(XGate(), Val(i))
            H .+= g .* convert_gates_to_matrix(n, Localised1SpinGate[a])
        end
    end

    return H
end


struct GenericBrickworkCircuit{T<:Real}
    nbits::Int
    nlayers::Int
    ngates::Int
    gate_angles::Matrix{T}
end
function GenericBrickworkCircuit(nbits, nlayers)
    ngates = sum(n->length((1 + (n-1) % 2):(nbits-1)), 1:nlayers)

    gate_array = zeros(Float64, 15, ngates);
    return GenericBrickworkCircuit(nbits, nlayers, ngates, gate_array)
end
function QuantumCircuits.apply!(ψ′, ψ, circuit::GenericBrickworkCircuit)
    gate_idx = 1
    for l in 1:circuit.nlayers
        for j in (1 + (l-1) % 2):(circuit.nbits-1)
            angles = view(circuit.gate_angles, :, gate_idx)
            gate = Localised2SpinAdjGate(build_general_unitary_gate(angles), Val(j))
            apply!(ψ′, ψ, gate)
            (ψ′, ψ) = (ψ, ψ′)
            gate_idx += 1
        end
    end
    return ψ
end


nbits = 4;
nlayers = 3;
J = 1;
h = 0.5;
g = 0;
H = build_hamiltonian(nbits, J, h, g);
@show H

circuit = GenericBrickworkCircuit(nbits, nlayers);
Random.randn!(circuit.gate_angles);
circuit.gate_angles .*= 0.01;

ψ = zero_state_tensor(nbits);
ψ′ = similar(ψ);
QuantumCircuits.apply!(ψ′, ψ, circuit);

ψ_vec = reshape(ψ, :, 1);
@assert norm(ψ_vec) ≈ 1
@show ψ_vec

measurement = adjoint(ψ_vec) * (H * ψ_vec)
measurement = real(measurement[begin])
@show measurement