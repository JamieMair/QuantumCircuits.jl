function apply(ψ, gate::AbstractGate)
    ψ′ = similar(ψ)
    return apply!(ψ′, ψ, gate)
end
function apply(ψ, gates::AbstractArray{<:AbstractGate}) 
    ψ′ = similar(ψ)
    if length(gates) == 1
        apply!(ψ′, ψ, g)
        return ψ′
    end
    
    ψ′′ = copy(ψ)
    for g in gates
        apply!(ψ′, ψ′′, g)
        ψ′′, ψ′ = (ψ′, ψ′′)
    end
    return ψ′′
end

function apply!(ψ′, ψ, gate::Localised1SpinGate)
    @boundscheck size(ψ′) == size(ψ) || error("ψ′ must be the same size as ψ.")
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)
    K = gate.target_gate_dim

    nqubits = ndims(ψ)
    @assert K <= nqubits && K >= 1 "The gate cannot be applied as it sits outside the qubit space."
    @inbounds for idxs in product((1:2 for _ in 1:ndims(ψ))...)
        pre_idxs = idxs[1:K-1]
        post_idxs = idxs[K+1:end]
        contract_idx = idxs[K]
        psi = ψ[idxs...]
        for i in 1:2
            ψ′[pre_idxs..., i, post_idxs...] += psi * u[i, contract_idx]
        end
    end
    ψ′
end
function apply!(ψ′, ψ, gate::Localised2SpinAdjGate)
    @boundscheck size(ψ′) == size(ψ) || error("ψ′ must be the same size as ψ.")
    ψ′ .= zero(eltype(ψ′))
    u = mat(gate)
    K = gate.target_gate_dim
    nqubits = ndims(ψ)
    @assert K < nqubits && K > 0 "The gate cannot be applied as it sits outside the qubit space."

    @inbounds for idxs in product((1:2 for _ in 1:ndims(ψ))...)
        pre_idxs = idxs[1:K-1]
        post_idxs = idxs[K+2:end]
        contract_idxs = (idxs[K], idxs[K+1])
        psi = ψ[idxs...]
        for j in 1:2
            for i in 1:2
                # u is reversed as Julia is column-major unlike row major of numpy
                ψ′[pre_idxs..., i, j, post_idxs...] += psi * u[i, j, contract_idxs...]
            end
        end
    end
    ψ′
end

@kernel function _apply_2spin!(ψ′, @Const(ψ), @Const(u), @Const(K), @Const(val_dims::Val{D})) where {D}
    idx = @index(Global, Linear)

    K = K
    idx = (idx-(1)) # change to zero based
    pre_index = idx % (1 << (K-1))
    i = (idx >> (K-1)) % 2 + 1
    j = (idx >> (K)) % 2 + 1
    post_index = (idx >> (K+1)) << (K+1)

    psi = zero(eltype(ψ))
    @inbounds for a in Base.OneTo(2)
        for b in Base.OneTo(2)
            # u is reversed as Julia is column-major unlike row major of numpy
            acc_idx = pre_index + ((a-1) << (K-1)) + ((b-1) << K) + post_index + 1 # change to one-based
            
            psi += ψ[acc_idx] * u[i, j, a, b]
        end
    end
    @inbounds ψ′[idx+1] = psi

    nothing
end

workgroup_default_size(::KernelAbstractions.CPU) = 16
workgroup_default_size(::KernelAbstractions.GPU) = 256

struct CPUApplyCache end
struct GPUApplyCache{CA, DA}
    cpu_gate_array_cache::CA
    device_gate_array_cache::DA
end

function construct_apply_cache(ψ)
    backend = get_backend(ψ)
    if typeof(backend) <: CPU
        return CPUApplyCache()
    else
        cpu_gate = Array{ComplexF64, 4}(undef, 2, 2, 2, 2)
        gpu_gate = allocate(backend, ComplexF64, 2, 2, 2, 2)
        return GPUApplyCache(cpu_gate, gpu_gate)
    end
end

function _apply_error_checks(ψ′, ψ, gate::Localised2SpinAdjGate)
    @assert get_backend(ψ′) == get_backend(ψ) "The backend of each array must be the same"
    @boundscheck size(ψ′) == size(ψ) || error("ψ′ must be the same size as ψ.")
    
    K = gate.target_gate_dim
    nqubits = ndims(ψ)
    @assert K < nqubits && K > 0 "The gate cannot be applied as it sits outside the qubit space."

    nothing
end

function apply!(::CPUApplyCache, ψ′::AbstractArray, ψ::AbstractArray, gate::Localised2SpinAdjGate)
    _apply_error_checks(ψ′, ψ, gate)
    return _apply!(ψ′, ψ, mat(gate), gate)
end
function apply!(cache::GPUApplyCache, ψ′::AbstractArray, ψ::AbstractArray, gate::Localised2SpinAdjGate)
    _apply_error_checks(ψ′, ψ, gate)

    # Copy over from static array onto the GPU
    cache.cpu_gate_array_cache .= mat(gate)
    u = cache.device_gate_array_cache
    device_backend = get_backend(u)
    KernelAbstractions.copyto!(device_backend, u, cache.cpu_gate_array_cache)


    return _apply!(ψ′, ψ, u, gate)
end

function _apply!(ψ′::AbstractArray{T, N}, ψ::AbstractArray{T, N}, u, gate::Localised2SpinAdjGate) where {T, N}
    ψ′ .= zero(eltype(ψ′))
    backend = get_backend(ψ′)
    kernel = _apply_2spin!(backend, min(workgroup_default_size(backend), length(ψ)))
    kernel(ψ′, ψ, u, gate.target_gate_dim, Val(N), ndrange=length(ψ))
    synchronize(backend)

    return ψ′
end

function _right_apply_gate!(M′, M, gate::Localised2SpinAdjGate)
    # Assuming M′ and M are both 2x2x....x2 tensors with nbits*2 dimensions
    fill!(M′, zero(eltype(M′)))
    u = mat(gate)
    K = gate.target_gate_dim
    nbits = ndims(M) ÷ 2
    X = K + nbits

    for idxs in product((1:2 for _ in 1:ndims(M))...)
        pre_idxs = idxs[1:X-1]
        post_idxs = idxs[X+2:end]
        contract_idxs = (idxs[X], idxs[X+1])
        m = M[idxs...]
        for j in 1:2
            for i in 1:2
                # u is reversed as Julia is column-major unlike row major of numpy
                M′[pre_idxs..., i, j, post_idxs...] += m * u[contract_idxs..., i, j]
            end
        end
    end
    return M′
end

function right_apply_gate!(M′::AbstractMatrix, M::AbstractMatrix, gate::Localised2SpinAdjGate)
    @boundscheck size(M′) == size(M) || error("M′ must be the same size as M")
    @boundscheck ndims(M) == 2 || error("Must be a matrix")
    @boundscheck size(M, 1) == size(M, 2) || error("Must be a square matrix")

    K = gate.target_gate_dim

    nbits = log2(size(M, 1))
    @assert nbits % 1 == 0 "The matrix must have a power of two edge"
    nbits = Int(round(nbits))

    M = reshape(M, Tuple(2 for _ in 1:2nbits))
    M′ = reshape(M′, Tuple(2 for _ in 1:2nbits))

    _right_apply_gate!(M′, M, gate)
end