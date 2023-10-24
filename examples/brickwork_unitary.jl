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
    # todo - correct this to use two buffers
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
    if ndims(ψ) != 2
        ψ = reshape(ψ, :, 1) # reshape to column vector
    end
    measurement = adjoint(ψ) * (H * ψ)
    return real(measurement[begin])
end
function measure(H::AbstractMatrix, ψ₀::AbstractArray, circuit::GenericBrickworkCircuit)
    ψ = copy(ψ₀)
    ψ′ = similar(ψ)
    ψ′′ = QuantumCircuits.apply!(ψ′, ψ, circuit);
    
    return measure(H, ψ′′)
end
function gradients(H::AbstractMatrix, ψ₀::AbstractArray, circuit::GenericBrickworkCircuit)
    gs = map(1:length(circuit.gate_angles)) do i
        l = reconstruct(circuit, i, π/2)
        r = reconstruct(circuit, i, -π/2)
        (measure(H, ψ₀, l)-measure(H, ψ₀, r)) / 2
    end

    return reshape(gs, size(circuit.gate_angles))
end
function optimise!(circuit::GenericBrickworkCircuit, H, ψ₀, epochs, lr)
    energies = Float64[]

    for i in 1:epochs+1        
        energy = measure(H, ψ₀, circuit)
        push!(energies, energy)

        # Gradients
        grad = gradients(H, ψ₀, circuit)

        circuit.gate_angles .-= lr .* grad
    end

    return energies
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

ψ₀ = zero_state_tensor(nbits);

epochs = 100
lr = 0.005
energies = optimise!(circuit, H, ψ₀, epochs, lr)

ψ = reshape(apply(ψ₀, circuit), :, 1);
# TODO: Add plots