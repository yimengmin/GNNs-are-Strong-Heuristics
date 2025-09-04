# GNNs-are-Strong-Heuristics

Got it 👍
Here’s a full **`README.md`** that documents your SLURM training workflow, checkpointing, and resume strategy — incorporating everything we discussed.

---

````markdown
# SCT-TSP Training on SLURM (7-Day Window, Checkpointing & Resume)

This README explains how to train the SCT-TSP model on an H100 GPU cluster with a 7-day time limit.  
It covers **job submission**, **checkpointing**, and **resuming training safely**.

---

## 1. Overview

- Training script: `trainingwithlstckpt.py` (supports checkpointing and resume).
- Scheduler: **SLURM**, with jobs limited to **7 days**.
- Strategy:
  - Save **checkpoints** regularly.
  - Catch SLURM’s `SIGTERM` signal to **safely save `last.ckpt` before timeout**.
  - Resume training from `last.ckpt` in the next job window until total epochs are reached.

---

## 2. SLURM Script Example

Save the following as `sbatchwithlstckpt.sh`:

```bash
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
````

### Submit the job

```bash
sbatch sbatchwithlstckpt.sh
```

### Monitor logs

```bash
tail -f Log/S100TrainHisShift-1L16Niter100Hid256Sct3adamWD0.0001Tau2.5.log
```

---

## 3. Checkpointing Strategy

Your script saves checkpoints in `SaveModels/`:

* **`last.ckpt`**
  The latest state (model, optimizer, RNG, history, best val length, step count).
  Updated periodically by:

  * `--save_every_epochs N` → every N epochs (e.g., `10` means save after epochs 10, 20, 30…).
  * `--save_every_steps N` → every N training steps (default off).
  * `--save_every_minutes M` → every M minutes (default off).

* **`best.ckpt`**
  Updated whenever validation improves.

* **Milestones**
  Every `--milestone_interval` epochs, saves the **best-so-far model** under a milestone filename.

### Why `SIGTERM` handling?

With:

```bash
#SBATCH --signal=B:TERM@180
```

SLURM sends `SIGTERM` **3 minutes before timeout**.
The trainer catches this signal and writes a final `last.ckpt`, ensuring you don’t lose progress.

---

## 4. Resuming Training

You can resume training across multiple 7-day windows until the target epoch count is reached.

### Auto-resume (recommended)

If `SaveModels/last.ckpt` exists:

```bash
python trainingwithlstckpt.py --save_dir SaveModels --auto_resume
```

No need to retype all hyperparameters — the script loads them from the checkpoint.

### Manual resume

```bash
python trainingwithlstckpt.py --resume SaveModels/last.ckpt
```

---

## 5. Hyperparameters & Resume Rules

### Must stay the same

Changing these will cause shape mismatches or optimizer state errors:

* `--hidden_dim`, `--n_layers`, `--sct_order`, `--gcn_order`
* `--n_iter`, `--tau`, `--tanh_scale`, `--noise_scale`
* `--shift`, `--distance_scale`
* `--optimizer` (if you want to reload optimizer state)

### Can be changed safely

* `--epochs` (total training length)
* `--batch_size`
* `--save_dir`, `--save_every_*`, `--milestone_interval`
* Logging options, visualization, patience, etc.

### Notes

* Optimizer & LR scheduler states are restored.

  * If you pass a new `--lr`, it will be ignored (state takes precedence).
  * To reset LR, you’d need to manually override after loading or skip optimizer state.
* RNG seeds are restored — reproducibility is preserved.
* Early stopping’s **best val** is restored, but patience counter resets.

---

## 6. Practical Tips

* **Safer saving**
  If 10 epochs is too risky, use:

  ```bash
  --save_every_epochs 5
  ```
* **Disk I/O**
  If checkpoints are large, you can rely on step/minute saves instead of very frequent epoch saves.
* **Multiple runs**
  Set `EPOCHS` to your total desired training length (e.g., 1500).
  If each SLURM job runs \~500 epochs before hitting the 7-day limit, just resubmit until all epochs are covered.
* **Switching optimizers mid-run**
  Possible, but you must avoid loading the old optimizer state. Safer to stick to one optimizer per experiment.

---

## 7. Troubleshooting

* **Job ended without saving**
  Check if your script actually traps `SIGTERM`. Without it, SLURM may kill the job without a final `last.ckpt`.
* **Backslashes in Bash**
  Ensure a backslash `\` is the very last character of a line (no trailing spaces).
* **Missing dirs**
  Run `mkdir -p Log SaveModels` before launching jobs.
* **Shape mismatch on resume**
  Likely caused by changing model hyperparameters between runs.

---

## 8. Example Workflow

**First job:**

```bash
sbatch sbatchwithlstckpt.sh
```

**Second job (after first ends at 7 days):**

```bash
sbatch sbatchwithlstckpt.sh
```

→ Thanks to `--auto_resume`, it continues from `SaveModels/last.ckpt`.

Repeat until `--epochs` are completed (e.g., 1500 total).

