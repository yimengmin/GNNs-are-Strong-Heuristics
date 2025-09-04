#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved main script for SCT TSP training with stable optimization
- Robust checkpointing / auto-resume (model/optimizer/RNG/history)
- Milestones: save the best-so-far model every N epochs
- Periodic saving by steps / minutes / epochs
- Safe SIGTERM preemption handler for clusters (e.g., SLURM/K8s)
- Optional mixed precision, warmup+cosine scheduler, early stopping

Quick start
-----------
# first run
python trainingwithlstckpt.py \
  --train_data data/tsp_100_uniform_train.pt \
  --val_data data/tsp_100_uniform_val.pt \
  --save_dir SaveModels \
  --epochs 600 --batch_size 512 \
  --use_scheduler --warmup_epochs 15 \
  --save_every_steps 1000 --save_every_minutes 30 --save_every_epochs 1 \
  --milestone_interval 50 --mixed_precision

# resume (auto-resume also works if last.ckpt exists in save_dir)
python sct_tsp_trainer.py --save_dir SaveModels --auto_resume

SLURM hint
----------
# send SIGTERM 3 minutes before timeout to trigger safe save
#SBATCH --signal=B:TERM@180
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import math
import numpy as np
import os
import time
import signal
from typing import Optional, Dict, Any

from cyclefastsags_stable import EnhancedUltraFastSCTGNN as GNN

from data_generator import SimpleTSPDataset, SimpleTSPDataLoader, load_tsp_dataset
from hardpermutation import to_exact_permutation_batched
from tsp_visualization import visualize_validation_tours, visualize_batch_tours
from utsploss import tsp_permutation_loss


# ----------------------------- Utility helpers -----------------------------

def set_seed(seed=42):
    """Set RNG seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Deterministic cuDNN is slower; enable only if needed.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_save(path: str, payload: Dict[str, Any]):
    """Atomic-ish save via temporary file + replace."""
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[Checkpoint] Saved: {path}")


# --------------------------- Schedulers and guards --------------------------

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing learning rate scheduler (epoch-based)."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            total = max(1, self.max_epochs - self.warmup_epochs)
            progress = (epoch - self.warmup_epochs) / total
            progress = min(max(progress, 0.0), 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


class EarlyStopping:
    """Early stopping utility based on validation loss/metric."""
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_value = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_value, model) -> bool:
        improved = val_value < self.best_value - self.min_delta
        if improved:
            self.best_value = val_value
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        should_stop = self.counter >= self.patience
        if should_stop and self.restore_best_weights and self.best_weights is not None:
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})
        return should_stop


class GradientClipper:
    """Adaptive gradient clipping based on recent gradient norm history."""
    def __init__(self, model, clip_percentile=10):
        self.model = model
        self.clip_percentile = clip_percentile
        self.grad_history = []

    def clip_gradients(self) -> float:
        norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                norms.append(p.grad.norm().item())
        if not norms:
            return 0.0
        current_norm = math.sqrt(sum(n ** 2 for n in norms))
        self.grad_history.append(current_norm)
        if len(self.grad_history) > 100:
            self.grad_history = self.grad_history[-100:]
        if len(self.grad_history) > 10:
            import numpy as _np
            clip_value = float(_np.percentile(self.grad_history, 100 - self.clip_percentile))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max(1e-6, clip_value))
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return current_norm


# --------------------------- Checkpoint helpers ----------------------------

def _rng_pack():
    return {
        'python': None,  # python random not used here; kept for parity
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
    }


def _rng_restore(state):
    if state is None:
        return
    if state.get('torch') is not None:
        torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if state.get('numpy') is not None:
        np.random.set_state(state['numpy'])


def _pack_ckpt(args, model, optimizer, epoch, global_step, best_val_length, history):
    return {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_length': float(best_val_length),
        'history': history,
        'args': vars(args),
        'rng_state': _rng_pack(),
        'save_time': time.time(),
    }


def _load_ckpt(path: str, map_location: str):
    return torch.load(path, map_location=map_location)


# ------------------------------- Optimizers --------------------------------

def create_optimizer(model, optimizer_type='adamw', lr=1e-3, weight_decay=1e-4):
    opt = optimizer_type.lower()
    if opt == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    elif opt == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'radam':
        try:
            from torch.optim import RAdam
            return RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        except Exception:
            print("RAdam not available, falling back to AdamW.")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay * 10)
        except Exception:
            print("Lion not available, falling back to AdamW.")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer '{optimizer_type}', using AdamW.")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# ------------------------------- Train / Val -------------------------------

def train_epoch(model, dataloader, optimizer, device, global_state: dict, shift=-1, distance_scale=5.0,
                grad_clipper: Optional[GradientClipper] = None, use_mixed_precision=False, loss_smoothing=0.0,
                saver: Optional["PeriodicSaver"] = None):
    """
    Train for one epoch with optional mixed precision / adaptive grad clip.
    Returns (mean_loss, mean_train_tour_length).
    """
    model.train()
    model.inference = False

    total_loss = 0.0
    total_tour_length = 0.0
    n_batches = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    smoothed_loss_val = None

    for coordinates_batch in tqdm(dataloader, desc="Training"):
        coordinates_batch = coordinates_batch.to(device)
        distance_matrices = torch.cdist(coordinates_batch, coordinates_batch)
        adjacency_matrices = torch.exp(-distance_matrices / distance_scale)
        node_features = coordinates_batch

        optimizer.zero_grad(set_to_none=True)

        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                P, logits = model(node_features, adjacency_matrices)
                tour_lengths, _ = tsp_permutation_loss(P, distance_matrices, shift)
                loss = tour_lengths.mean()
            scaler.scale(loss).backward()
            if grad_clipper:
                scaler.unscale_(optimizer)
                grad_clipper.clip_gradients()
            scaler.step(optimizer)
            scaler.update()
        else:
            P, logits = model(node_features, adjacency_matrices)
            tour_lengths, _ = tsp_permutation_loss(P, distance_matrices, shift)
            loss = tour_lengths.mean()
            loss.backward()
            if grad_clipper:
                grad_clipper.clip_gradients()
            optimizer.step()

        # Logging smoothing (for stability only)
        if smoothed_loss_val is None:
            smoothed_loss_val = loss.item()
        else:
            alpha = float(loss_smoothing)
            smoothed_loss_val = alpha * loss.item() + (1.0 - alpha) * smoothed_loss_val

        total_loss += loss.item()
        total_tour_length += tour_lengths.mean().item()
        n_batches += 1

        # Step counter + periodic save
        global_state['global_step'] += 1
        if saver is not None:
            saver.maybe_save(global_state)

        # NaN / explosion guard
        if not math.isfinite(loss.item()) or loss.item() > 1e6:
            print("Warning: loss became non-finite or exploded; breaking out of epoch.")
            break

    mean_loss = total_loss / max(1, n_batches)
    mean_train_len = total_tour_length / max(1, n_batches)
    return mean_loss, mean_train_len


def validate_epoch(model, dataloader, device, shift=-1, distance_scale=5.0, epoch=None, visualize=False):
    """Validate with hard permutation decoding (Hungarian on logits)."""
    model.eval()
    model.inference = True

    total_val_len = 0.0
    n_batches = 0
    first_batch_coords = None
    first_batch_heatmaps = None

    with torch.no_grad():
        for bidx, coordinates_batch in enumerate(tqdm(dataloader, desc="Validation")):
            coordinates_batch = coordinates_batch.to(device)

            distance_matrices = torch.cdist(coordinates_batch, coordinates_batch)
            adjacency_matrices = torch.exp(-distance_matrices / distance_scale)
            node_features = coordinates_batch

            logits = model(node_features, adjacency_matrices)
            P_hard = to_exact_permutation_batched(logits)
            val_lengths, heatmaps = tsp_permutation_loss(P_hard, distance_matrices, shift)
            total_val_len += val_lengths.mean().item()
            n_batches += 1

            if bidx == 0 and visualize and epoch is not None:
                first_batch_coords = coordinates_batch.detach().cpu()
                first_batch_heatmaps = heatmaps.detach().cpu()

    if visualize and epoch is not None and first_batch_coords is not None:
        try:
            visualize_validation_tours(first_batch_coords, first_batch_heatmaps, epoch)
        except Exception as e:
            print(f"Validation visualization failed: {e}")

    return total_val_len / max(1, n_batches)


# ---------------------------- Periodic saver --------------------------------

class PeriodicSaver:
    def __init__(self, args, model, optimizer, save_dir: str):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.save_dir = ensure_dir(save_dir)
        self.last_time = time.time()

    def path(self, name: str):
        return os.path.join(self.save_dir, name)

    def save_last(self, epoch: int, global_step: int, best_val_length: float, history: dict):
        payload = _pack_ckpt(self.args, self.model, self.optimizer, epoch, global_step, best_val_length, history)
        _atomic_save(self.path('last.ckpt'), payload)

    def maybe_save(self, global_state: dict):
        # step-based
        steps = int(self.args.save_every_steps)
        if steps > 0 and global_state['global_step'] % steps == 0:
            self.save_last(global_state['epoch'], global_state['global_step'], global_state['best_val'], global_state['history'])
            self.last_time = time.time()
            return
        # time-based
        minutes = float(self.args.save_every_minutes)
        if minutes > 0 and (time.time() - self.last_time) >= minutes * 60:
            self.save_last(global_state['epoch'], global_state['global_step'], global_state['best_val'], global_state['history'])
            self.last_time = time.time()


# ---------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description='Improved SCT TSP Training with Stable Optimization and Milestone Saving')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Data
    parser.add_argument('--train_data', type=str, default='data/tsp_100_uniform_train.pt')
    parser.add_argument('--val_data', type=str, default='data/tsp_100_uniform_val.pt')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--sct_order', type=int, default=4)
    parser.add_argument('--gcn_order', type=int, default=2)
    parser.add_argument('--tanh_scale', type=float, default=40.0)
    parser.add_argument('--tau', type=float, default=5.0)
    parser.add_argument('--n_iter', type=int, default=60)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--shift', type=int, default=-1, help='Shift for rolling transpose (negative integer)')
    parser.add_argument('--distance_scale', type=float, default=5.0, help='Adjacency: exp(-dist/scale)')

    # Training
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--device', type=str, default='cuda')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'radam', 'lion'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=15)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    # Stability
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--adaptive_grad_clip', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--loss_smoothing', type=float, default=0.0)

    # Visualization
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--viz_freq', type=int, default=20)

    # Saving / resume
    parser.add_argument('--save_dir', type=str, default='SaveModels')
    parser.add_argument('--resume', type=str, default='', help='Path to a checkpoint to resume from')
    parser.add_argument('--auto_resume', action='store_true', help='Auto-resume from save_dir/last.ckpt if present')
    parser.add_argument('--save_every_steps', type=int, default=0, help='Periodic last.ckpt every N steps (0=off)')
    parser.add_argument('--save_every_minutes', type=float, default=0.0, help='Periodic last.ckpt every N minutes (0=off)')
    parser.add_argument('--save_every_epochs', type=int, default=1, help='Save last.ckpt every N epochs (>=1)')

    # Milestone saving
    parser.add_argument('--milestone_interval', type=int, default=50, help='Save best-so-far model every N epochs')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    save_dir = ensure_dir(args.save_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_coords, _ = load_tsp_dataset(args.train_data)
    val_coords, _ = load_tsp_dataset(args.val_data)

    print("Creating datasets (coordinates only)...")
    train_dataset = SimpleTSPDataset(train_coords)
    val_dataset = SimpleTSPDataset(val_coords)

    train_loader = SimpleTSPDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = SimpleTSPDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Build model
    input_dim = 2
    output_dim = train_coords.shape[1]  # number of cities
    print("Building SCT model...")
    print(f"Input dim: {input_dim}, Output dim (cities): {output_dim}")
    print(f"Shift: {args.shift}, Distance scale: {args.distance_scale}")

    model = GNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        n_layers=args.n_layers,
        sct_order=args.sct_order,
        gcn_order=args.gcn_order,
        tanh_scale=args.tanh_scale,
        tau=args.tau,
        n_iter=args.n_iter,
        noise_scale=args.noise_scale,
        inference_mode=False
    ).to(device)

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and optional scheduler
    optimizer = create_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    print(f"Optimizer: {args.optimizer.upper()} (weight_decay={args.weight_decay})")

    scheduler = None
    if args.use_scheduler:
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr
        )
        print(f"Scheduler: Warmup({args.warmup_epochs}) + Cosine to min_lr={args.min_lr}")

    # Optional early stopping and gradient clipping
    early_stopping = EarlyStopping(patience=args.patience) if args.early_stopping else None
    if early_stopping:
        print(f"EarlyStopping enabled (patience={args.patience})")

    grad_clipper = GradientClipper(model) if args.adaptive_grad_clip else None
    if grad_clipper:
        print("Adaptive gradient clipping enabled")

    if args.mixed_precision:
        print("Mixed precision training enabled")

    # --- Resume logic ---
    start_epoch = 0
    best_val_length = float('inf')
    history = {'train_loss': [], 'train_length': [], 'val_length': [], 'lr': []}
    global_state = {'epoch': 0, 'global_step': 0, 'best_val': best_val_length, 'history': history}

    last_ckpt_path = os.path.join(save_dir, 'last.ckpt')
    auto_resume_used = False

    if args.resume:
        ckpt_path = args.resume
    elif args.auto_resume and os.path.isfile(last_ckpt_path):
        ckpt_path = last_ckpt_path
        auto_resume_used = True
    else:
        ckpt_path = ''

    if ckpt_path:
        print(f"[Resume] Loading checkpoint from: {ckpt_path}")
        ckpt = _load_ckpt(ckpt_path, map_location=str(device))
        try:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # move optimizer state tensors to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            _rng_restore(ckpt.get('rng_state'))
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            global_state['global_step'] = int(ckpt.get('global_step', 0))
            history = ckpt.get('history', history)
            best_val_length = float(ckpt.get('best_val_length', float('inf')))
            global_state['history'] = history
            global_state['best_val'] = best_val_length
            print(f"[Resume] start_epoch={start_epoch}, global_step={global_state['global_step']}, best_val={best_val_length:.2f}")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint due to: {e}. Starting fresh.")
            start_epoch = 0
            best_val_length = float('inf')
            global_state['global_step'] = 0

    # Periodic saver & SIGTERM handler
    saver = PeriodicSaver(args, model, optimizer, save_dir)

    def _sigterm_handler(sig, frame):
        print(f"[Signal] Caught {sig}. Saving last.ckpt and exiting...")
        saver.save_last(global_state['epoch'], global_state['global_step'], global_state['best_val'], global_state['history'])
        os._exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Options: scheduler={bool(scheduler)}, early_stop={bool(early_stopping)}, "
          f"grad_clip={bool(grad_clipper)}, mixed_precision={args.mixed_precision}")
    print(f"Milestone interval: every {args.milestone_interval} epochs (best-so-far)")

    # Best payload holder (for milestone saves)
    best_payload_so_far = None

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        global_state['epoch'] = epoch

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            current_lr = scheduler.step(epoch)

        train_loss, train_len = train_epoch(
            model, train_loader, optimizer, device,
            global_state=global_state,
            shift=args.shift,
            distance_scale=args.distance_scale,
            grad_clipper=grad_clipper,
            use_mixed_precision=args.mixed_precision,
            loss_smoothing=args.loss_smoothing,
            saver=saver,
        )

        # Validation
        should_visualize = args.visualize and ((epoch + 1) % args.viz_freq == 0)
        val_len = validate_epoch(
            model, val_loader, device,
            shift=args.shift,
            distance_scale=args.distance_scale,
            epoch=epoch,
            visualize=should_visualize
        )

        # Bookkeeping
        history.setdefault('train_loss', []).append(train_loss)
        history.setdefault('train_length', []).append(train_len)
        history.setdefault('val_length', []).append(val_len)
        history.setdefault('lr', []).append(current_lr)

        print(f"Epoch {epoch+1:4d} | LR={current_lr:.2e} | "
              f"Train Loss={train_loss:.4f} | Train Len={train_len:.2f} | Val Len={val_len:.2f}")

        # Save last every N epochs
        if (epoch + 1) % max(1, args.save_every_epochs) == 0:
            saver.save_last(epoch, global_state['global_step'], best_val_length, history)

        # Best checkpoint update
        if val_len < best_val_length:
            best_val_length = val_len
            global_state['best_val'] = best_val_length
            best_payload_so_far = _pack_ckpt(args, model, optimizer, epoch, global_state['global_step'], best_val_length, history)
            _atomic_save(os.path.join(save_dir, 'best.ckpt'), best_payload_so_far)
            print(f"★ New best validation length: {val_len:.2f}")

        # Milestone saving: every N epochs, save the best-so-far payload under a milestone filename
        if (epoch + 1) % max(1, args.milestone_interval) == 0 and best_payload_so_far is not None:
            milestone_path = (
                f"milestone_best_up_to_epoch_{epoch+1}_"
                f"size_{output_dim}_hidden_{args.hidden_dim}_"
                f"{args.optimizer}_tau_{args.tau}_"
                f"n_iter_{args.n_iter}_noise_{args.noise_scale}_"
                f"shift_{args.shift}_dist_scale_{args.distance_scale}_"
                f"n_layers{args.n_layers}_"
                f"seed_{args.seed}.ckpt"
            )
            _atomic_save(os.path.join(save_dir, milestone_path), best_payload_so_far)

        # Early stopping check
        if early_stopping and early_stopping(val_len, model):
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best validation length: {best_val_length:.2f}")
            saver.save_last(epoch, global_state['global_step'], best_val_length, history)
            break

        # Instability warning
        if len(history['val_length']) > 10:
            recent = history['val_length'][-10:]
            if max(recent) - min(recent) > 100.0:
                print("Warning: Validation variance is high; consider reducing LR or enabling gradient clipping.")

    print("\nTraining completed.")
    print(f"Best validation length: {best_val_length:.2f}")
    print("Training history embedded in the latest best/last checkpoints.")

    # Optional: plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history['train_loss']) + 1)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        ax1.plot(epochs_range, history['train_loss']); ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.grid(True)
        ax2.plot(epochs_range, history['train_length'], label='Train'); ax2.plot(epochs_range, history['val_length'], label='Validation'); ax2.set_title('Tour Lengths'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Tour Length'); ax2.legend(); ax2.grid(True)
        ax3.plot(epochs_range, history['lr']); ax3.set_title('Learning Rate'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('LR'); ax3.set_yscale('log'); ax3.grid(True)
        ax4.plot(epochs_range, history['val_length']); ax4.set_title('Validation Length (Detailed)'); ax4.set_xlabel('Epoch'); ax4.set_ylabel('Tour Length'); ax4.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"training_curves_shift_{args.shift}_dist_scale_{args.distance_scale}_seed_{args.seed}.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training curves: {plot_path}")
    except ImportError:
        print("matplotlib not available; skipping curve plotting.")


if __name__ == "__main__":
    main()

