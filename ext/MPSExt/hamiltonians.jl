function QuantumCircuits.MPSTFIMHamiltonian(nbits, J, h, g)
    ham = MPSHamiltonian(nbits)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    ZZ = kron(Z, Z)

    if J != 0
        for i in 1:nbits-1
            add!(ham, MPSTerm(i, -J * ZZ))
        end
    end
    if g != 0
        for i in 1:nbits
            add!(ham, MPSTerm(i, (-J * g) * X))
        end
    end
    if h != 0
        for i in 1:nbits
            add!(ham, MPSTerm(i, (-J * h) * Z))
        end
    end

    return ham
end