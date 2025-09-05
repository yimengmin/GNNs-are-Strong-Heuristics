#!/bin/bash
#SBATCH --qos=low
#SBATCH --signal=B:TERM@180 # send SIGTERM 3 minutes before timeout
#SBATCH --job-name=ddp_4xH100
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH -p full
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32


# NCCL & threading hints
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=8


# Hyper-params
EPOCHS=1500
SIZE=100
NITER=100
HID=256
NLAYER=16
SFT=-1
SCT=3
OPT=adam
WD=0.0001
TAU=2.6
BS_PER_GPU=512 # per-GPU batch size (global batch = 4 * 128 = 512)


mkdir -p Log SaveModels
LOGFILE=Log/ddp4_S${SIZE}_H${HID}_L${NLAYER}_N${NITER}_opt${OPT}_wd${WD}_tau${TAU}_$(date +%F_%H%M).log
exec > "$LOGFILE" 2>&1


nvidia-smi || true
python -V || true


# Launch DDP on 4 GPUs (single node)
torchrun --standalone --nproc_per_node=4 trainingwithlstckpt_ddp.py \
--ddp \
--train_data /mnt/beegfs/bulk/mirror/ym499/UTSPHardTSP100Accereate/data/tsp_${SIZE}_uniform_train.pt \
--val_data /mnt/beegfs/bulk/mirror/ym499/UTSPHardTSP100Accereate/data/tsp_${SIZE}_uniform_val.pt \
--shift ${SFT} \
--distance_scale 5.0 --tau ${TAU} \
--hidden_dim ${HID} \
--sct_order ${SCT} \
--n_iter ${NITER} \
--optimizer ${OPT} \
--n_layers ${NLAYER} \
--batch_size ${BS_PER_GPU} \
--weight_decay ${WD} \
--adaptive_grad_clip \
--use_scheduler --warmup_epochs 15 \
--early_stopping --patience 50 \
--epochs ${EPOCHS} \
--save_dir SaveModels \
--auto_resume \
--save_every_epochs 10


echo "[INFO] Finished at $(date)"
