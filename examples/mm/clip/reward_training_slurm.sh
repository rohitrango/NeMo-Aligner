#!/bin/bash
#SBATCH -A coreai_dlalgo_genai
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=coreai_dlalgo_genai-rewardtraining:*
#SBATCH --partition=batch
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --dependency=singleton

## Multijob setup (no need to provide anything)
export NNODES=$SLURM_JOB_NUM_NODES
if [ "$NNODES" -eq 1 ]; then
    # Parameters for single node
    echo "Running on a single node"
    DISTRIBUTED_PARAMS="--master_port=30030"
else
    # Parameters for multiple nodes
    echo "Running on $NNODES nodes, initializing rendezvous point"
    head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    export RDZV_ID=$RANDOM
    echo Node IP: $head_node_ip
    # write actual params
	DISTRIBUTED_PARAMS="--rdzv_id $RDZV_ID --rdzv_backend c10d --rdzv_endpoint $head_node_ip:30030"
fi

# change this if using other modes
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:=24}

echo "Running script with additional kwargs $ADDITIONAL_KWARGS"

srun --container-image /lustre/fsw/coreai_dlalgo_genai/rohit/draft_container.sqsh \
    --container-mounts /lustre/fsw/coreai_dlalgo_genai/rohit/NeMo-Aligner/:/opt/nemo-aligner,/lustre/fsw/coreai_dlalgo_genai/rohit/NeMo:/opt/NeMo,/lustre/fsw/coreai_dlalgo_genai/rohit/megatron-lm:/opt/megatron-lm \
    bash /opt/nemo-aligner/examples/mm/clip/reward_training.sh

# torchrun --nproc-per-node=8 $DISTRIBUTED_PARAMS --nnodes=$NNODES /opt/nemo-aligner/examples/mm/clip/train_reward_model.py \
# model.micro_batch_size=$MICRO_BATCH_SIZE trainer.devices=8 trainer.num_nodes=$NNODES exp_manager.create_wandb_logger=True $ADDITIONAL_KWARGS

