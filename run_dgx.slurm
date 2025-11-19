#!/bin/bash
#SBATCH --job-name=fathomnet-4gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate fathomnet

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --config config/experiment-01.yaml
