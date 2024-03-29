cd(@__DIR__)


if !isdir("hpc")
    mkdir("hpc")
end

if !isdir("hpc/output")
    mkdir("hpc/output")
end

if !isdir("hpc/results")
    mkdir("hpc/results")
end

if !isdir("hpc/scripts")
    mkdir("hpc/scripts")
    open("hpc/scripts/example.sh", "w") do fw
        write(fw,
            raw"""
            #!/bin/bash

            #SBATCH --ntasks=10
            #SBATCH --cpus-per-task=2
            #SBATCH --mem-per-cpu=4096
            #SBATCH -o hpc/output/test_%j.out

            module purge
            module load julia/1.10.2

            wd=$(pwd)
            run_file="$wd/hpc/test.jl"
            julia --project $run_file

            rm julia-*-*-*.out
            """)
    end
end