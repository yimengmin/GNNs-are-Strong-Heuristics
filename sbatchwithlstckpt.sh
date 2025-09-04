#!/bin/bash
#SBATCH --qos=low
#SBATCH --signal=B:TERM@180        # send SIGTERM 3 min before timeout
#SBATCH --job-name=longjob100
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH -p full
#SBATCH --gres=gpu:h100:1

# ---------------- Parameters ----------------
EPOCHS=1500
SIZE=100
NITER=100
HID=256
NLAYER=16
SFT=-1
SCT=3
OPT=adam                  # optimizer: adam | adamw | radam | lion
WD=0.0001
tau=2.5

# ---------------- Logging -------------------
mkdir -p Log SaveModels
LOGFILE=Log/S${SIZE}TrainHisShift${SFT}L${NLAYER}Niter${NITER}Hid${HID}Sct${SCT}${OPT}WD${WD}Tau${tau}.log
exec > "${LOGFILE}" 2>&1

echo "[INFO] Job started at $(date)"
nvidia-smi || true
python -V || true

# ---------------- Training ------------------
python -u trainingwithlstckpt.py \
  --train_data /mnt/beegfs/bulk/mirror/ym499/UTSPHardTSP100Accereate/data/tsp_${SIZE}_uniform_train.pt \
  --val_data   /mnt/beegfs/bulk/mirror/ym499/UTSPHardTSP100Accereate/data/tsp_${SIZE}_uniform_val.pt \
  --shift ${SFT} \
  --distance_scale 5.0 --tau ${tau} \
  --hidden_dim ${HID} \
  --sct_order ${SCT} \
  --n_iter ${NITER} \
  --adaptive_grad_clip \
  --use_scheduler \
  --optimizer ${OPT} \
  --n_layers ${NLAYER} \
  --batch_size 512 \
  --weight_decay ${WD} \
  --early_stopping --patience 50 \
  --epochs ${EPOCHS} \
  --save_dir SaveModels \
  --auto_resume \
  --save_every_epochs 10

echo "[INFO] Job finished at $(date)"
