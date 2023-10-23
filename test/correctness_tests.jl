using TestItems

@testitem "Test Cat state generation" begin
    using LinearAlgebra
    
    for nbits in (4, 5, 7)
        ψ = zero_state_tensor(nbits)
        ψ′ = copy(ψ)
        hadamard = Localised1SpinGate(HadamardGate(), Val(1))
        apply!(ψ′, ψ, hadamard)
        (ψ, ψ′) = (ψ′, ψ)

        for i in 1:(nbits-1)
            cnot = Localised2SpinAdjGate(CNOTGate(), Val(i))
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
        hadamard = Localised1SpinGate(HadamardGate(), Val(1))

        ψ = convert_gates_to_matrix(nbits, [hadamard]) * ψ;
        for i in 1:(nbits-1)
            cnot = Localised2SpinAdjGate(CNOTGate(), Val(i))
            ψ = convert_gates_to_matrix(nbits, [cnot]) * ψ;
        end

        ψ_expected = similar(ψ)
        ψ_expected .= 0
        ψ_expected[begin] = 1
        ψ_expected[end] = 1
        ψ_expected ./= norm(ψ_expected)

        @test ψ ≈ ψ_expected
    end
end