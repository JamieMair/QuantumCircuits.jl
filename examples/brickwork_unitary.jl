using Revise
includet("test_brickwork_problem.jl")

    
nbits = 4;
nlayers = 6;
J = 1;
h = 0.5;
g = 0;
H = build_hamiltonian(nbits, J, h, g);

circuit = GenericBrickworkCircuit(nbits, nlayers);
Random.randn!(circuit.gate_angles);
circuit.gate_angles .*= 0.01;

ψ₀ = zero_state_tensor(nbits);

epochs = 100
lr = 0.01
energies = optimise!(circuit, H, ψ₀, epochs, lr);

ψ = reshape(apply(ψ₀, circuit), :, 1);
ψ = sqrt.(real.(ψ .* conj.(ψ)))

eigen_decomp = eigen(H);
min_energy = minimum(eigen_decomp.values);
ground_state = eigen_decomp.vectors[:, findfirst(x->x==min_energy, eigen_decomp.values)]

using Plots
using LaTeXStrings
begin
    plt = plot(0:(length(energies)-1), energies, label=L"\langle H \ \rangle", lw=2, color=:black)
    hline!(plt, [min_energy], label=L"E_0", linestyle=:dash, lw=2)
    xlabel!(plt, "Epochs")
    ylabel!(plt, "TFIM Energy")
    xlims!(plt, (0, epochs))
    plt
end