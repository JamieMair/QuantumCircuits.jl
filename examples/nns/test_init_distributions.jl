using Flux

function test_network(nangles, s = 0, c = 0.01)
    if s == 0
        eff_gain = c * sqrt((nangles + 1) / 2)
        return Chain(Dense(1=>nangles, identity; bias=false, init=Flux.glorot_normal(gain=eff_gain)))
    else
        eff_gain = c * sqrt((nangles + s)) / 2
        return Chain(
            Dense(1=>s, Flux.tanh; init=Flux.glorot_normal),
            Dense(s=>s, Flux.tanh; init=Flux.glorot_normal),
            Dense(s=>s, Flux.tanh; init=Flux.glorot_normal),
            Dense(s=>nangles, identity; init=Flux.glorot_normal(gain=eff_gain)),
        )
    end

end

function test_distribution(nangles, s, repeats=1024)
    angles = Float32[]

    for _ in 1:repeats
        n = test_network(nangles, s)
        ys = n(1:1)
        append!(angles, ys)
    end

    return angles
end

function plot_test_distributions(ys)
        f = Figure()
        ax = Axis(f[1,1], xlabel="σ", ylabel="Freq")
        for y in ys
            hist!(ax, y, alpha=0.05, normalization=:pdf)
        end
        # f[1,2] = Legend(f, ax)
        return f
end