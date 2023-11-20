using Random;
include("../test_brickwork_problem.jl")

function gate_gradient(nbits, nlayers, H, ψ₀, index=1)
    circuit = GenericBrickworkCircuit(nbits, nlayers)
    Random.rand!(circuit.gate_angles)
    circuit.gate_angles .*= 2 * π
    return gradient(H, ψ₀, circuit, index)
end

function git_sha()
    out=IOBuffer()
    run(pipeline(`git rev-parse HEAD`, stdout=out))
    hash = strip(String(take!(out)))
    return hash
end

function run_trial(config::Dict{Symbol, Any}, trial_id) 
    results = Dict{Symbol, Any}()
    nbits = config[:nbits]
    nlayers = config[:nlayers]
    nrepeats = config[:nrepeats]
    gate_index = config[:gate_index]
    J = config[:J]
    h = config[:h]
    g = config[:g]

    seed = Int(Random.rand(UInt16))
    results[:seed] = seed
    Random.seed!(seed)

    results[:git_sha] = git_sha()

    H = build_hamiltonian(nbits, J, h, g);
    ψ₀ = zero_state_tensor(nbits);

    results[:gradients] = map(1:nrepeats) do _
        return gate_gradient(nbits, nlayers, H, ψ₀, gate_index)
    end

    return results
end