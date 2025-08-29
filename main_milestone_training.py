"""
Improved main script for SCT TSP training with stable optimization
Adds milestone saving: save the best-so-far model every 50 epochs
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import math
import numpy as np
import os

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
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def atomic_save_checkpoint(path: str, payload: dict):
    """
    Save a checkpoint payload to disk; ensure parent directory exists.
    This wrapper can be extended to use temporary files for atomic renames.
    """
    ensure_dir(os.path.dirname(path) or "SaveModels")
    torch.save(payload, path)
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
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine decay
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

        # Keep a short recent history to avoid stale scales
        if len(self.grad_history) > 100:
            self.grad_history = self.grad_history[-100:]

        if len(self.grad_history) > 10:
            clip_value = float(np.percentile(self.grad_history, 100 - self.clip_percentile))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max(1e-6, clip_value))
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return current_norm


# ------------------------------- Optimizers --------------------------------

def create_optimizer(model, optimizer_type='adamw', lr=1e-3, weight_decay=1e-4):
    """Create optimizer with reasonable defaults."""
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
            # Lion often uses smaller LR and larger weight decay scales
            return Lion(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay * 10)
        except Exception:
            print("Lion not available, falling back to AdamW.")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"Unknown optimizer '{optimizer_type}', using AdamW.")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# ------------------------------- Train / Val -------------------------------

def train_epoch(model, dataloader, optimizer, device, shift=-1, distance_scale=5.0,
                grad_clipper: GradientClipper = None, use_mixed_precision=False, loss_smoothing=0.0):
    """
    Train for one epoch with optional mixed precision and adaptive gradient clipping.
    Returns (mean_loss, mean_train_tour_length).
    """
    model.train()
    model.inference = False

    total_loss = 0.0
    total_tour_length = 0.0
    n_batches = 0

    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    smoothed_loss_val = None

    for coordinates_batch in tqdm(dataloader, desc="Training"):
        coordinates_batch = coordinates_batch.to(device)

        # Pairwise Euclidean distances
        distance_matrices = torch.cdist(coordinates_batch, coordinates_batch)

        # Adjacency via distance-decay kernel
        adjacency_matrices = torch.exp(-distance_matrices / distance_scale)

        # Node features: raw (x, y)
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

        # Optional exponential smoothing (for logging stability only)
        if smoothed_loss_val is None:
            smoothed_loss_val = loss.item()
        else:
            alpha = float(loss_smoothing)
            smoothed_loss_val = alpha * loss.item() + (1.0 - alpha) * smoothed_loss_val

        total_loss += loss.item()
        total_tour_length += tour_lengths.mean().item()
        n_batches += 1

        # NaN / explosion guard
        if not math.isfinite(loss.item()) or loss.item() > 1e6:
            print("Warning: loss became non-finite or exploded; breaking out of epoch.")
            break

    mean_loss = total_loss / max(1, n_batches)
    mean_train_len = total_tour_length / max(1, n_batches)
    return mean_loss, mean_train_len


def validate_epoch(model, dataloader, device, shift=-1, distance_scale=5.0, epoch=None, visualize=False):
    """
    Validate with hard permutation decoding (Hungarian on logits).
    Returns mean validation tour length.
    """
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

            # Forward to get logits for Hungarian decoding
            logits = model(node_features, adjacency_matrices)

            # Hard permutation via Hungarian
            P_hard = to_exact_permutation_batched(logits)

            # Compute tour lengths from hard permutations
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

    # Milestone saving
    parser.add_argument('--milestone_interval', type=int, default=50, help='Save best-so-far model every N epochs')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    ensure_dir('SaveModels/')
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

    # Training bookkeeping
    best_val_length = float('inf')
    best_payload_so_far = None  # Will hold the best-so-far model payload for milestone saves
    history = {'train_loss': [], 'train_length': [], 'val_length': [], 'lr': []}

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Options: scheduler={bool(scheduler)}, early_stop={bool(early_stopping)}, "
          f"grad_clip={bool(grad_clipper)}, mixed_precision={args.mixed_precision}")
    print(f"Milestone interval: every {args.milestone_interval} epochs (best-so-far)")

    # Training loop
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            current_lr = scheduler.step(epoch)

        train_loss, train_len = train_epoch(
            model, train_loader, optimizer, device,
            shift=args.shift,
            distance_scale=args.distance_scale,
            grad_clipper=grad_clipper,
            use_mixed_precision=args.mixed_precision,
            loss_smoothing=args.loss_smoothing
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
        history['train_loss'].append(train_loss)
        history['train_length'].append(train_len)
        history['val_length'].append(val_len)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1:4d} | LR={current_lr:.2e} | "
              f"Train Loss={train_loss:.4f} | Train Len={train_len:.2f} | Val Len={val_len:.2f}")

        # If current val is best, update the best payload and write a "best" checkpoint
        if val_len < best_val_length:
            best_val_length = val_len
            best_path = (f"SaveModels/best_stable_sct_model_"
                         f"size_{output_dim}_hidden_{args.hidden_dim}_"
                         f"{args.optimizer}_tau_{args.tau}_"
                         f"n_iter_{args.n_iter}_noise_{args.noise_scale}_"
                         f"shift_{args.shift}_dist_scale_{args.distance_scale}_"
                         f"n_layers{args.n_layers}_"
                         f"seed_{args.seed}.pt")

            best_payload_so_far = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_length': val_len,
                'training_history': history,
                'args': args
            }
            atomic_save_checkpoint(best_path, best_payload_so_far)
            print(f"★ New best validation length: {val_len:.2f}")

        # Milestone saving: every N epochs, save the best-so-far payload under a milestone filename
        if (epoch + 1) % max(1, args.milestone_interval) == 0 and best_payload_so_far is not None:
            milestone_path = (f"SaveModels/milestone_best_up_to_epoch_{epoch+1}_"
                              f"size_{output_dim}_hidden_{args.hidden_dim}_"
                              f"{args.optimizer}_tau_{args.tau}_"
                              f"n_iter_{args.n_iter}_noise_{args.noise_scale}_"
                              f"shift_{args.shift}_dist_scale_{args.distance_scale}_"
                              f"n_layers{args.n_layers}_"
                              f"seed_{args.seed}.pt")
            atomic_save_checkpoint(milestone_path, best_payload_so_far)

        # Early stopping check
        if early_stopping and early_stopping(val_len, model):
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best validation length: {best_val_length:.2f}")
            break

        # Simple instability heuristic (optional warning)
        if len(history['val_length']) > 10:
            recent = history['val_length'][-10:]
            if max(recent) - min(recent) > 100.0:
                print("Warning: Validation variance is high; consider reducing learning rate or enabling gradient clipping.")

    print("\nTraining completed.")
    print(f"Best validation length: {best_val_length:.2f}")
    print("Training history embedded in the latest best checkpoint.")

    # Optional: plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history['train_loss']) + 1)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        ax1.plot(epochs_range, history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(epochs_range, history['train_length'], label='Train')
        ax2.plot(epochs_range, history['val_length'], label='Validation')
        ax2.set_title('Tour Lengths')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Tour Length')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(epochs_range, history['lr'])
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.grid(True)

        ax4.plot(epochs_range, history['val_length'])
        ax4.set_title('Validation Length (Detailed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Tour Length')
        ax4.grid(True)

        plt.tight_layout()
        plot_path = f"training_curves_shift_{args.shift}_dist_scale_{args.distance_scale}_seed_{args.seed}.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training curves: {plot_path}")
    except ImportError:
        print("matplotlib not available; skipping curve plotting.")


if __name__ == "__main__":
    main()

