#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=coreai_dlalgo_genai-draft2:*
#SBATCH --partition=polar3
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --overcommit

export INF_STEPS=${INF_STEPS:=25}
export KL_COEF=${KL_COEF:=0.1}
# export KL_COEF=0.1
export LR=${LR:=0.00025}
# export LR=0.0001
export ETA=${ETA:=0.0}
export DATASET=${DATASET:="pickapic50k.tar"}
export MICRO_BS=${MICRO_BS:=1}
export GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=4}
export PEFT=${PEFT:="sdlora"}
export LOG_WANDB=${LOG_WANDB:="True"}
export JOBNAME=${JOBNAME:="default"}
# export SLEEP=${SLEEP:=10}

# srun --container-image gitlab-master.nvidia.com/dl/joc/nemo-ci/train:pipe.13548649 --container-mounts /lustre/fsw/portfolios/coreai/users/rohitkumarj/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/portfolios/coreai/users/rohitkumarj/NeMo:/opt/NeMo,/lustre/fsw/portfolios/coreai/users/rohitkumarj/megatron-lm:/opt/megatron-lm bash /opt/nemo-aligner/examples/mm/stable_diffusion/launch_draft_xl.sh
srun --container-image /lustre/fsw/portfolios/coreai/users/rohitkumarj/draft_container.sqsh --container-mounts \
/lustre/fsw/portfolios/coreai/users/rohitkumarj/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/portfolios/coreai/users/rohitkumarj/NeMo:/opt/NeMo,/lustre/fsw/portfolios/coreai/users/rohitkumarj/megatron-lm:/opt/megatron-lm  \
bash /opt/nemo-aligner/examples/mm/stable_diffusion/launch_draft_xl.sh
