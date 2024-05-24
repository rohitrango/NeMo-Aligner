#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=coreai_dlalgo_llm-dpo:*
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton

export LR=${LR:=0.000025}
export ACTOR_MICRO_BS=${ACTOR_MICRO_BS:=1}
export GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=64}

srun --container-image gitlab-master.nvidia.com/dl/joc/nemo-ci/train:pipe.12668828 --container-mounts /lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/coreai_dlalgo_genai/ataghibakhsh/NeMo:/opt/NeMo bash /opt/nemo-aligner/examples/mm/stable_diffusion/launch_dpo.sh
