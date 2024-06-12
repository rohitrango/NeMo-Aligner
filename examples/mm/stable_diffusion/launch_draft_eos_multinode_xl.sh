#!/bin/bash
#SBATCH -A coreai_dlalgo_genai
#SBATCH -p batch
#SBATCH -N 4
#SBATCH -t 4:00:00
#SBATCH --ntasks=4
#SBATCH --job-name=coreai_dlalgo_genai-draft:*
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton

export INF_STEPS=${INF_STEPS:=25}
export KL_COEF=${KL_COEF:=0.1}
export LR=${LR:=0.00025}
export ETA=${ETA:=0.0}
export DATASET=${DATASET:="pickapic50k.tar"}
export MICRO_BS=${MICRO_BS:=1}
export GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=4}
export PEFT=${PEFT:="sdlora"}
export LOG_WANDB=${LOG_WANDB:="True"}
export JOBNAME=${JOBNAME:="default"}
export CONFIG_NAME=${CONFIG_NAME:="draftp_sdxl"}

## garbage SBATCH --ntasks-per-node=1

### Multinode setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export RDZV_ID=$RANDOM

echo Node IP: $head_node_ip

# srun --container-image gitlab-master.nvidia.com/dl/joc/nemo-ci/train:pipe.13548649 --container-mounts /lustre/fsw/coreai_dlalgo_genai/rohit/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/coreai_dlalgo_genai/rohit/NeMo:/opt/NeMo,/lustre/fsw/coreai_dlalgo_genai/rohit/megatron-lm:/opt/megatron-lm bash /opt/nemo-aligner/examples/mm/stable_diffusion/launch_draft_xl.sh
# srun --container-image /lustre/fsw/portfolios/coreai/users/rohitkumarj/draft_container.sqsh --container-mounts /lustre/fsw/coreai_dlalgo_genai/rohit/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/coreai_dlalgo_genai/rohit/NeMo:/opt/NeMo,/lustre/fsw/coreai_dlalgo_genai/rohit/megatron-lm:/opt/megatron-lm bash /opt/nemo-aligner/examples/mm/stable_diffusion/launch_draft_xl.sh

srun --ntasks-per-node=8 --container-image /lustre/fsw/coreai_dlalgo_genai/rohit/draft_container.sqsh --container-mounts /lustre/fsw/coreai_dlalgo_genai/rohit/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/coreai_dlalgo_genai/rohit/NeMo:/opt/NeMo,/lustre/fsw/coreai_dlalgo_genai/rohit/megatron-lm:/opt/megatron-lm bash /opt/nemo-aligner/examples/mm/stable_diffusion/launch_draft_xl.sh

