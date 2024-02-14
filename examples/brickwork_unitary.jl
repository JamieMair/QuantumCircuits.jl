using Revise
using QuantumCircuits
include("test_brickwork_problem.jl")


nbits = 12;
nlayers = 8;
J = 1.0;
g = 0.5;
H = TFIMHamiltonian(J, g);
H_matrix = build_hamiltonian(nbits, J, g);

circuit = GenericBrickworkCircuit(nbits, nlayers);
nrepeats = 3

epochs = 80
lr = 0.01

function test_optimise(circuit, H, epochs, lr)
    Random.randn!(circuit.gate_angles)
    circuit.gate_angles .*= 0.01

    ψ₀ = zero_state_tensor(nbits)
    energies = optimise!(circuit, H, ψ₀, epochs, lr)
    return energies
end

energy_trajectories = [test_optimise(circuit, H, epochs, lr) for _ in 1:nrepeats];


ψ₀ = zero_state_tensor(nbits);
ψ = reshape(apply(ψ₀, circuit), :, 1);
ψ = sqrt.(real.(ψ .* conj.(ψ)))

eigen_decomp = eigen(H_matrix);
min_energy = minimum(eigen_decomp.values);
ground_state = eigen_decomp.vectors[:, findfirst(x -> x == min_energy, eigen_decomp.values)]

using CairoMakie
using LaTeXStrings
begin
    f = Figure()
    ax = Axis(f[1, 1],
        title="Grad. Descent on TFIM with $(nlayers) layers and $(nbits) sites.",
        xlabel="# Epochs",
        ylabel="<E>")
    for energies in energy_trajectories
        lines!(ax, 0:(length(energies)-1), energies, label=L"\langle H \ \rangle", color=:black, alpha=0.7)
    end
    hlines!(ax, [min_energy], label=L"E_0", linestyle=:dash)
    xlims!(ax, (0, epochs))
    f
end