#!/bin/bash -l
#SBATCH --job-name=t2t-padding
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=results/padding-%j.log

set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the Touch2Touch repository root}"

if (( $# > 1 )) || { (( $# == 1 )) && ! [[ "$1" =~ ^([1-9]|[1-3][0-9]|4[0-6])$ ]]; }; then
    printf 'Usage: sbatch scripts/compare_padding.sbatch [samples per tool, 1-46]\n' >&2
    exit 2
fi
pilot=()
if (( $# == 1 )); then
    pilot=(--samples-per-tool "$1")
fi

module load conda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
output="results/padding-${SLURM_JOB_ID:?}"

conda run --no-capture-output -n touch2touch-icp-clean \
    python -Bu -m scripts.compare_icp --dataset padding --preflight-only \
    --image-metrics --examples-per-tool 1

limit="$(squeue -h -j "$SLURM_JOB_ID" -o '%L')"
conda run --no-capture-output -n touch2touch-smoke \
    python -Bu -m scripts.compare_icp --dataset padding --generate-predictions \
    --time-left "$limit" --output "$output/predictions" "${pilot[@]}"

conda run --no-capture-output -n touch2touch-icp-clean \
    python -Bu -m scripts.compare_icp --dataset padding --predictions "$output/predictions" \
    --image-metrics --examples-per-tool 1 --output "$output/evaluation" "${pilot[@]}"
