#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=coreai_dlalgo_genai-draft2:*
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton

export INF_STEPS=${INF_STEPS:=25}
export KL_COEF=${KL_COEF:=0.2}
export LR=${LR:=0.00025}
export ETA=${ETA:=0.0}
export DATASET=${DATASET:="pickapic50k.tar"}
export MICRO_BS=${MICRO_BS:=2}
export GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=4}
export PEFT=${PEFT:="sdlora"}

srun --container-image /lustre/fsw/coreai_dlalgo_genai/rohit/draft_container.sqsh  --container-mounts /lustre/fsw/coreai_dlalgo_genai/rohit/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/coreai_dlalgo_genai/rohit/NeMo:/opt/NeMo,/lustre/fsw/coreai_dlalgo_genai/rohit/megatron-lm:/opt/megatron-lm bash /opt/nemo-aligner/examples/mm/stable_diffusion/launch_draft.sh
