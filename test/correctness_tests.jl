using TestItems

@testitem "Test Cat state generation" begin
    using LinearAlgebra
    
    for nbits in (4, 5, 7)
        ψ = zero_state_tensor(nbits)
        ψ′ = copy(ψ)
        hadamard = Localised1SpinGate(HadamardGate(), 1)
        apply!(ψ′, ψ, hadamard)
        (ψ, ψ′) = (ψ′, ψ)

        for i in 1:(nbits-1)
            cnot = Localised2SpinAdjGate(CNOTGate(), i)
            apply!(ψ′, ψ, cnot)
            (ψ, ψ′) = (ψ′, ψ)
        end

        ψ_expected = similar(ψ)
        ψ_expected .= 0
        ψ_expected[begin] = 1
        ψ_expected[end] = 1
        ψ_expected ./= norm(ψ_expected)

        @test ψ ≈ ψ_expected
    end
end


@testitem "Test Cat state generation with matrices" begin
    using LinearAlgebra
    
    for nbits in (4, 5, 7)
        ψ = zero_state_vec(nbits)
        hadamard = Localised1SpinGate(HadamardGate(), 1)

        ψ = convert_gates_to_matrix(nbits, QuantumCircuits.AbstractGate[hadamard]) * ψ;
        for i in 1:(nbits-1)
            cnot = Localised2SpinAdjGate(CNOTGate(), i)
            ψ = convert_gates_to_matrix(nbits, QuantumCircuits.AbstractGate[cnot]) * ψ;
        end

        ψ_expected = similar(ψ)
        ψ_expected .= 0
        ψ_expected[begin] = 1
        ψ_expected[end] = 1
        ψ_expected ./= norm(ψ_expected)

        @test ψ ≈ ψ_expected
    end
end

@testitem "Random single qubit operators" begin
    using LinearAlgebra
    using Random

    function generate_random_unitary()
        while true
            u = randn(ComplexF64, (2,2))
            d = det(u)
            if d != 0
                return u ./ d
            end
        end
    end

    nbits = 5
    # Create a random initial state
    ψ = randn(ComplexF64, Tuple(2 for _ in 1:nbits))
    ψ ./= norm(ψ)

    
    gates = map(1:nbits) do i
        Localised1SpinGate(Generic1SpinGate(generate_random_unitary()), i)
    end

    ψ_vec = reshape(ψ, :, 1)
    equiv_operator = convert_gates_to_matrix(nbits, gates)
    ψ_expected = reshape(equiv_operator * ψ_vec, size(ψ))

    ψ′ = apply(ψ, gates)
    @test ψ′ ≈ ψ_expected
end


@testitem "Random brickwork" begin
    using LinearAlgebra
    using Random

    function generate_random_2spin_gate_mat()
        while true
            u = randn(ComplexF64, (4,4))
            d = det(u)
            if d != 0
                return reshape(u ./ d, (2,2,2,2))
            end
        end
    end

    nbits = 5
    # Create a random initial state
    ψ = randn(ComplexF64, Tuple(2 for _ in 1:nbits))
    ψ ./= norm(ψ)

    nlayers = 3
    for layer in 1:nlayers
        start_index = 1 + (layer - 1) % 2
        end_index = (nbits-1)
        gates = map(start_index:2:end_index) do i
            Localised2SpinAdjGate(Generic2SpinGate(generate_random_2spin_gate_mat()), i)
        end

        ψ_vec = reshape(ψ, :, 1)
        equiv_operator = convert_gates_to_matrix(nbits, gates)
        ψ_expected = reshape(equiv_operator * ψ_vec, size(ψ))

        ψ′ = apply(ψ, gates)
        @test ψ′ ≈ ψ_expected

        if layer != nlayers
            ψ .= ψ′ # Set ψ for the next layer
        end
    end
end

@testitem "East Model Hamiltonian" begin
    using QuantumCircuits
    using Random
    using LinearAlgebra
    s = -0.1
    c = 0.1
    A = -exp(-s) * sqrt(c * (1-c))
    B = (1 - 2c)

    nbits = 5
    ψ = randn(ComplexF64, Tuple(2 for _ in 1:nbits))
    ψ ./= norm(ψ)

    H_eff = EastModelHamiltonian(c, s)
    @test H_eff.A ≈ A
    @test H_eff.B ≈ B
    # Build H
    H = QuantumCircuits.mat(H_eff, nbits);

    E_eff = measure(H_eff, ψ)
    E_actual = measure(H, ψ)


    @test E_eff ≈ E_actual
end