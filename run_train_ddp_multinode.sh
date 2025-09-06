#!/bin/bash
#SBATCH --job-name=ddp_2x4
#SBATCH --qos=low
#SBATCH --partition=full
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4          # or a100:4 if that’s what the node actually has
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@180

# ====== EDIT THESE TWO LINES TO MATCH YOUR SETUP ======
WORKDIR="/mnt/beegfs/bulk/mirror/ym499/CAM-GNN-S500"
SCRIPT="trainingwithlstckpt_ddp.py"
#SCRIPT="trainingwithlstckpt.py"     # or trainingwithlstckpt_ddp.py
# ======================================================

# Log + sanity
mkdir -p Log SaveModels
LOGFILE=Log/ddp8_S500_H128_L24_N120_optadam_wd0.0001_tau3.0_$(date +%F_%H%M).log
exec > "$LOGFILE" 2>&1

echo "[INFO] Host: $(hostname)"
echo "[INFO] SLURM_NODELIST=$SLURM_NODELIST"
echo "[INFO] Using WORKDIR=$WORKDIR, SCRIPT=$SCRIPT"
if [[ ! -d "$WORKDIR" ]]; then
  echo "[FATAL] WORKDIR does not exist: $WORKDIR"
  exit 2
fi
if [[ ! -f "$WORKDIR/$SCRIPT" ]]; then
  echo "[FATAL] Script not found: $WORKDIR/$SCRIPT"
  ls -la "$WORKDIR"
  exit 2
fi

nvidia-smi || true
python -V || true

# NCCL / networking
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=8
# If your cluster uses a specific NIC: export NCCL_SOCKET_IFNAME=ib0

# Rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# Train hyperparams (edit as needed)
EPOCHS=1000
SIZE=500
NITER=120
HID=256
NLAYER=24
SFT=-1
SCT=3
OPT=adam
WD=0.0001
TAU=3.0
BS_PER_GPU=64
NUM_WORKERS=4

echo "[INFO] Starting multi-node torchrun at $(date)"

# One launcher per node; each launcher spawns 4 local ranks
srun --ntasks-per-node=1 bash -lc "
  set -e
  cd '$WORKDIR'
  echo '[INFO] CWD on node' \$(hostname): \$(pwd)
  echo '[INFO] Listing script:'; ls -l '$SCRIPT'
  export PYTHONUNBUFFERED=1
  export PYTHONPATH='$WORKDIR':\$PYTHONPATH

  torchrun \
    --nnodes=\${SLURM_NNODES} \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    '$SCRIPT' \
      --ddp \
      --num_workers ${NUM_WORKERS} \
      --train_data /mnt/beegfs/bulk/mirror/ym499/UTSPHard/data/tsp_${SIZE}_uniform_train_2m.pt \
      --val_data   /mnt/beegfs/bulk/mirror/ym499/UTSPHard/data/tsp_${SIZE}_uniform_val.pt \
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
      --save_dir SaveModels_S${SIZE}NIter${n_iter}NL${n_layers}EPS${epochs}Tau${TAU}HID${HID}OPT${optimizer} --auto_resume \
      --save_every_epochs 10
"

echo "[INFO] Finished at $(date)"


