#!/bin/bash
#SBATCH --qos=low
#SBATCH --job-name=longjob100
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH -p full
#SBATCH --gres=gpu:h100:1

# Parameters
SIZE=100
NITER=100
HID=256
NLAYER=16
SFT=-1
SCT=3
OPT=adam   # <-- add optimizer here
WD=0.0001
tau=3.0
# Build log filename
LOGFILE=Log/S${SIZE}TrainHisShift${SFT}L${NLAYER}Niter${NITER}Hid${HID}Sct${SCT}${OPT}WD${WD}Tau${tau}.log

# Redirect stdout and stderr
exec > "${LOGFILE}" 2>&1

# Run training
python -u main.py \
  --train_data ../UTSPHardTSP100Accereate/data/tsp_${SIZE}_uniform_train.pt \
  --val_data   ../UTSPHardTSP100Accereate/data/tsp_${SIZE}_uniform_val.pt \
  --shift ${SFT} \
  --distance_scale 5.0 --tau ${tau} \
  --hidden_dim ${HID} \
  --sct_order ${SCT} \
  --n_iter ${NITER} \
  --adaptive_grad_clip \
  --use_scheduler \
  --optimizer ${OPT}  \
  --n_layers ${NLAYER} \
  --batch_size 512 \
  --weight_decay ${WD} \
  --early_stopping --patience 50 

