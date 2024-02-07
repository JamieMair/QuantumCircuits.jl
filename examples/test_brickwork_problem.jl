using QuantumCircuits
using Random
using LinearAlgebra
using ProgressBars


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