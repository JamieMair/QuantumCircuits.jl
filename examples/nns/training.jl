using Flux
using ProgressBars
include("logging.jl")

function extract_hyperparameters(network, config)
    hparams = Dict{String, Any}()
    nparams = sum(length, Flux.params(network[begin:end-2]))
    hparams["nparams"] = nparams

    keys = [:J, :h, :g, :epochs, :nbits, :nlayers, :learning_rate, :weight_init_magnitude]

    for k in keys
        hparams[string(k)] = config[k]
    end
    return hparams
end

function train!(network, epochs, logger=NullLogger(); lr=0.01, use_gpu::Bool = true, use_progress::Bool=false, log_info_freq::Int=1)
    input = use_gpu ? [1.0f0;;] |> Flux.gpu : [1.0f0;;];
    
    losses = Float32[];
    info = Dict{Symbol, Any}()
    grad_info = Dict{Int, Any}()
    optim = Flux.setup(Flux.Adam(lr), network);
    iter = use_progress ? ProgressBar(1:epochs) : (1:epochs)
    for epoch in iter
        energy, grads = Flux.withgradient(network) do m 
            m(input)
        end

        if (epoch-1) % log_info_freq == 0
            # Calculate gradient statistics
            flat_grads, _ = Flux.destructure(grads |> Flux.cpu)
            n_grads = length(flat_grads)
            mean_abs_grad = mean(abs, flat_grads)
            norm_grads = norm(flat_grads)
            mean_norm_grads = norm_grads / sqrt(n_grads)
            
            
            log_scalar!(logger, "grads/norm_grads", epoch, norm_grads)
            log_scalar!(logger, "grads/mean_norm_grads", epoch, mean_norm_grads)
            log_scalar!(logger, "grads/mean_abs_grad", epoch, mean_abs_grad)

            grad_info[epoch] = (;
                n_grads=n_grads,
                mean_abs_grad=mean_abs_grad,
                norm_grads=norm_grads,
                mean_norm_grads=mean_norm_grads
            )
        end

        log_scalar!(logger, "metrics/energy", epoch, energy)

        Flux.update!(optim, network, grads[1])
        push!(losses, energy)
    end


    info[:gradient_info] = grad_info

    return losses, info
end