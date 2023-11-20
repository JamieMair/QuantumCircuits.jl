using Revise
include("test_brickwork_problem.jl")


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

using CairoMakie
using LaTeXStrings
begin
    f = Figure()
    ax = Axis(f[1,1],
        title="Grad. Descent on TFIM with $(nlayers) layers and $(nbits) sites.",
        xlabel="# Epochs",
        ylabel="<E>")

    
    lines!(ax, 0:(length(energies)-1), energies, label=L"\langle H \ \rangle", color=:black)
    hlines!(ax, [min_energy], label=L"E_0", linestyle=:dash)
    xlims!(ax, (0, epochs))
    f
end