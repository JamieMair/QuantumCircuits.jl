using QuantumCircuits
using LinearAlgebra
using SparseArrays

import KrylovKit: eigsolve

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

function build_sparse_tfim_hamiltonian(n, J, h, g)

    matrix_eltype = promote_type(typeof(J), typeof(h), typeof(g))
    Jh = J*h
    Jg = J * g

    diagonal_elements = map(0:((2^n)-1)) do C
        zz = zero(matrix_eltype)
        for i in 0:(n-2)
            r = unsafe_trunc(UInt8, C >> i) % 0b0100
            # Matching spins
            zz += (r == 0b11) || (r == 0b00)
            # Different spins
            zz -= (r == 0b01) || (r == 0b10)
        end

        z = (n - 2*count_ones(C))

        return -(J * zz + Jh * z)
    end
    H = spdiagm(diagonal_elements)

    for i in 1:2^n
        Ci = i - 1
        for k in 0:(n-1)
            Cj = xor(Ci, 1 << k)
            H[Cj + 1, i] += -Jg
        end
    end

    return LinearAlgebra.Symmetric(H)
end


function find_tfim_ground_state(nbits, J, h, g)
    H = build_sparse_tfim_hamiltonian(nbits, J, h, g)
    eigen_vals, eigen_vecs, _ = eigsolve(H, 2^nbits, 1, :SR)
    

    return first(eigen_vals), first(eigen_vecs)
end