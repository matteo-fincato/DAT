#!/bin/bash
#SBATCH --job-name=DAT_small_panoptic_band
#SBATCH --output=/mnt/beegfs/work/ToyotaHPE/logs/DAT/small_panoptic_band_train_out.txt
#SBATCH --error=/mnt/beegfs/work/ToyotaHPE/logs/DAT/small_panoptic_band_train_err.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --account=ricerca_generica
#SBATCH --partition=prod
##SBATCH --mem=16GB
#SBATCH --exclude huber,lurcanio,pippobaudo
#SBATCH --time 20:00:00
##SBATCH --dependency=afterany:1922466

source activate dat
#gervasoni,pippobaudo,rezzonico

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST

echo "${nodelist[*]}"
module unload cuda
module load cuda/11.0

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
# export NCCL_BLOCKING_WAIT=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512


srun --label python -m torch.distributed.run \
--rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
--nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE \
main.py \
--output /mnt/beegfs/work/ToyotaHPE/code/3D_HPE/DAT/output \
--cfg configs/dat_small_panoptic.yaml \
--tag small_panoptic_band \
--train_split train \
--train_split_json train \
--val_split val \
--val_split_json val &

wait