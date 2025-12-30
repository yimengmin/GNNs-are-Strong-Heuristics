#!/bin/bash
#SBATCH --job-name=ddp_3x4
#SBATCH --qos=low
#SBATCH --partition=full
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4          # or a100:4 if that’s what the node actually has
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@180

# ====== EDIT THESE TWO LINES TO MATCH YOUR SETUP ======
WORKDIR="/mnt/beegfs/bulk/mirror/ym499/harmonics-AttenGNNs-are-Strong-Heuristics"
SCRIPT="trainingwithlstckpt_ddp_bf16.py"


## data: tsp_500_uniform_train_6m_bf16.pt
#SCRIPT="trainingwithlstckpt.py"     # or trainingwithlstckpt_ddp.py
# ======================================================

# Train hyperparams (edit as needed)
EPOCHS=1000
SIZE=500
NITER=100
HID=384
NLAYER=48
SFT=-1
SCT=3
OPT=adam
WD=0.0001
#WD=0.000067
TAU=3.5
BS_PER_GPU=50 # 42 for l=60
NUM_WORKERS=4
DISSCALE=5.0
DPOUT=0.5
#LR=0.006 # 0.002*3 for multigpu training, Learning Rate scaling, for reference, set 2e-3 for bs=512
LR=0.002 # 0.002

# Log + sanity
mkdir -p Log SaveModels
LOGFILE=Log/BF16_ddp16_S${SIZE}_H${HID}_L${NLAYER}_N${NITER}_opt${OPT}_wd${WD}_dpout${DPOUT}_tau${TAU}_DS${DISSCALE}_$(date +%F_%H%M).log

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
      --train_data /mnt/beegfs/bulk/mirror/ym499/UTSPHard/data/tsp_${SIZE}_uniform_train_5m.pt \
      --val_data   /mnt/beegfs/bulk/mirror/ym499/UTSPHard/data/tsp_${SIZE}_uniform_val.pt \
      --shift ${SFT} \
      --distance_scale ${DISSCALE} --tau ${TAU} \
      --hidden_dim ${HID} \
      --sct_order ${SCT} \
      --n_iter ${NITER} \
      --optimizer ${OPT} \
      --n_layers ${NLAYER} \
      --batch_size ${BS_PER_GPU} \
      --weight_decay ${WD} \
      --adaptive_grad_clip \
      --use_scheduler --warmup_epochs 15 \
      --early_stopping --patience 100 \
      --epochs ${EPOCHS} \
      --save_dir SaveModels/BF16S${SIZE}NIter${NITER}NL${NLAYER}EPS${EPOCHS}Tau${TAU}HID${HID}OPT${OPT}DS${DISSCALE}DP${DPOUT} \
      --auto_resume \
      --save_every_epochs 5 \
      --dropout ${DPOUT} \
      --lr ${LR} 
"

echo "[INFO] Finished at $(date)"


