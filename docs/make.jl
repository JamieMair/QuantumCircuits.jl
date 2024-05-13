using QuantumCircuits
using Documenter

QuantumCircuits.install_mps_support()
QuantumCircuits.init_mps_support()

DocMeta.setdocmeta!(QuantumCircuits, :DocTestSetup, :(using QuantumCircuits); recursive=true)

makedocs(;
    modules=[QuantumCircuits],
    authors="Jamie Mair <JamieMair@users.noreply.github.com> and contributors",
    repo="https://github.com/JamieMair/QuantumCircuits.jl/blob/{commit}{path}#{line}",
    sitename="QuantumCircuits.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JamieMair.github.io/QuantumCircuits.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JamieMair/QuantumCircuits.jl",
    devbranch="main",
)
