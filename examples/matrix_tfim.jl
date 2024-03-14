using QuantumCircuits


function build_hamiltonian(n, J, h, g)
    H = zeros(2^n, 2^n)
    for i in 1:n-1
        a = Localised1SpinGate(ZGate(), i)
        b = Localised1SpinGate(ZGate(), i+1)
        H .+= convert_gates_to_matrix(n, Localised1SpinGate[a, b])
    end

    if g != 0
        for i in 1:n
            a = Localised1SpinGate(XGate(), i)
            H .+= g .* convert_gates_to_matrix(n, Localised1SpinGate[a])
        end
    end
    
    if h != 0
        for i in 1:n
            a = Localised1SpinGate(ZGate(), i)
            H .+= h .* convert_gates_to_matrix(n, Localised1SpinGate[a])
        end
    end

    H .*= (-J)

    return H
end