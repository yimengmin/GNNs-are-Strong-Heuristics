"""
Improved main script for SCT TSP training with stable optimization
Uses advanced optimization techniques to reduce training variance
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import math
import numpy as np

# Import SCT model
#from fastsags import UltraFastSCTGNN as GNN

#from cyclefastsags import EnhancedUltraFastSCTGNN as GNN
from cyclefastsags_stable import EnhancedUltraFastSCTGNN as GNN

from data_generator import SimpleTSPDataset, SimpleTSPDataLoader, load_tsp_dataset
from hardpermutation import to_exact_permutation_batched
from tsp_visualization import visualize_validation_tours, visualize_batch_tours
import os

import wandb

def setup_wandb_auth():
    """Setup WandB authentication with helpful instructions"""
    print("\n" + "="*60)
    print("WANDB AUTHENTICATION SETUP")
    print("="*60)
    print("To use WandB logging, you need to authenticate. Choose one option:")
    print()
    print("1. Interactive login (recommended for first time):")
    print("   wandb login")
    print()
    print("2. Set environment variable:")
    print("   export WANDB_API_KEY=your_api_key_here")
    print("   # Add to ~/.bashrc or ~/.zshrc for persistence")
    print()
    print("3. Create ~/.netrc file:")
    print("   echo 'machine api.wandb.ai' >> ~/.netrc")
    print("   echo 'login user' >> ~/.netrc") 
    print("   echo 'password your_api_key_here' >> ~/.netrc")
    print("   chmod 600 ~/.netrc")
    print()
    print("4. Set in code (not recommended for security):")
    print("   wandb.login(key='your_api_key_here')")
    print()
    print("Get your API key from: https://wandb.ai/authorize")
    print("="*60)


# Ensure deterministic behavior
def set_seed(seed=42):
    """Set seeds for reproducible results"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # Slower but more deterministic
    # torch.backends.cudnn.benchmark = False

# Advanced Learning Rate Schedulers
class WarmupCosineScheduler:
    """Warmup + Cosine Annealing scheduler"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict({k: v.to(model.device if hasattr(model, 'device') else 'cuda') 
                                     for k, v in self.best_weights.items()})
            return True
        return False

class GradientClipper:
    """Adaptive gradient clipping"""
    def __init__(self, model, clip_percentile=10):
        self.model = model
        self.clip_percentile = clip_percentile
        self.grad_history = []
        
    def clip_gradients(self):
        # Collect gradient norms
        grad_norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if not grad_norms:
            return 0.0
            
        current_norm = math.sqrt(sum(norm**2 for norm in grad_norms))
        self.grad_history.append(current_norm)
        
        # Keep only recent history
        if len(self.grad_history) > 100:
            self.grad_history = self.grad_history[-100:]
        
        # Adaptive clipping based on gradient history
        if len(self.grad_history) > 10:
            clip_value = np.percentile(self.grad_history, 100 - self.clip_percentile)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
        return current_norm

# Simple version
if not os.path.exists('SaveModels/'):
    os.makedirs('SaveModels/')
from utsploss import tsp_permutation_loss


def create_optimizer(model, optimizer_type='adamw', lr=1e-3, weight_decay=1e-4):
    """Create optimizer with better defaults"""
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, 
                           betas=(0.9, 0.999), eps=1e-8)

#    if optimizer_type.lower() == 'adamw':
#        return optim.AdamW(
#            model.parameters(),
#            lr=lr,
#            weight_decay=weight_decay,
#            betas=(0.9, 0.999),
#            eps=1e-8,
#            foreach=True,   # use multi-tensor CUDA kernels
#            fused=False      # enable fused CUDA kernels (requires recent PyTorch + CUDA)
#        )
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'radam':
        try:
            from torch.optim import RAdam
            return RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("RAdam not available, using AdamW instead")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr*0.1, weight_decay=weight_decay*10)  # Lion uses different scales
        except ImportError:
            print("Lion not available, using AdamW instead")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_epoch(model, dataloader, optimizer, device, shift=-1, distance_scale=5.0, 
                grad_clipper=None, use_mixed_precision=False, loss_smoothing=0.0):
    """Enhanced training epoch with stability improvements"""
    model.train()
    #model = torch.compile(model, mode='max-autotune')   ### does not work? qaq..
    model.inference = False
    
    total_loss = 0
    total_tour_length = 0
    num_batches = 0
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Loss smoothing for stability
    smoothed_loss = None
    
    for coordinates_batch in tqdm(dataloader, desc="Training"):
        coordinates_batch = coordinates_batch.to(device)
        batch_size, n_cities = coordinates_batch.shape[:2]
        
        # Compute distance matrices
        distance_matrices = torch.cdist(coordinates_batch, coordinates_batch)
        
        # Apply distance scaling factor
        adjacency_matrices = torch.exp(-1 * distance_matrices / distance_scale)
        
        # Node features are just coordinates (2 features)
        node_features = coordinates_batch  # (B, N, 2)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                P, logits = model(node_features, adjacency_matrices)
                tour_lengths, heatmap = tsp_permutation_loss(P, distance_matrices, shift)
                loss = tour_lengths.mean()
        else:
            P, logits = model(node_features, adjacency_matrices)
            tour_lengths, heatmap = tsp_permutation_loss(P, distance_matrices, shift)
            loss = tour_lengths.mean()
        
        # Loss smoothing for stability, not used in this paper
        if smoothed_loss is None:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = loss_smoothing * loss.item() + (1 - loss_smoothing) * smoothed_loss
        
        # Backward pass
        if use_mixed_precision:
            scaler.scale(loss).backward()
            
            # Gradient clipping with mixed precision
            if grad_clipper:
                scaler.unscale_(optimizer)
                grad_clipper.clip_gradients()
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if grad_clipper:
                grad_clipper.clip_gradients()
            
            optimizer.step()
        
        total_loss += loss.item()
        total_tour_length += tour_lengths.mean().item()
        num_batches += 1
        
        # Early stopping if loss explodes
        if loss.item() > 1e6:
            print("Warning: Loss explosion detected!")
            break
    
    return total_loss / num_batches, total_tour_length / num_batches

def validate_epoch(model, dataloader, device, shift=-1, distance_scale=5.0, 
                   epoch=None, visualize=False):
    """Validate using hard permutation matrices"""
    model.eval()
    model.inference = True
    
    total_tour_length = 0
    num_batches = 0
    first_batch_coords = None
    first_batch_heatmaps = None
    
    with torch.no_grad():
        for batch_idx, coordinates_batch in enumerate(tqdm(dataloader, desc="Validation")):
            coordinates_batch = coordinates_batch.to(device)
            batch_size, n_cities = coordinates_batch.shape[:2]
            
            # Compute distance matrices
            distance_matrices = torch.cdist(coordinates_batch, coordinates_batch)
            
            # Apply distance scaling factor
            adjacency_matrices = torch.exp(-1 * distance_matrices / distance_scale)
            
            # Node features are just coordinates (2 features)
            node_features = coordinates_batch  # (B, N, 2)
            
            # Get logits for Hungarian algorithm
            logits = model(node_features, adjacency_matrices)
            
            # Convert to hard permutation matrices
            P_hard = to_exact_permutation_batched(logits)
            
            # Compute tour lengths with hard assignments
            tour_lengths, val_heatmaps = tsp_permutation_loss(P_hard, distance_matrices, shift)
            
            total_tour_length += tour_lengths.mean().item()
            num_batches += 1
            
            # Save first batch for visualization
            if batch_idx == 0 and visualize and epoch is not None:
                first_batch_coords = coordinates_batch.cpu()
                first_batch_heatmaps = val_heatmaps.cpu()
    
    # Visualize tours from first batch
    if visualize and epoch is not None and first_batch_coords is not None:
        try:
            visualize_validation_tours(first_batch_coords, first_batch_heatmaps, epoch)
        except Exception as e:
            print(f"Visualization failed: {e}")

    return total_tour_length / num_batches

def main():
    parser = argparse.ArgumentParser(description='Improved SCT TSP Training with Stable Optimization')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    # WandB parameters
#    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
#    parser.add_argument('--wandb_project', type=str, default='multiscale-graph-tsp', help='WandB project name')
#    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
#    parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')
#    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[], help='WandB tags')
#    parser.add_argument('--wandb_offline', action='store_true', help='Run WandB in offline mode')
#
    
    # Data parameters
    parser.add_argument('--train_data', type=str, default='data/tsp_100_uniform_train.pt')
    parser.add_argument('--val_data', type=str, default='data/tsp_100_uniform_val.pt')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--sct_order', type=int, default=4)
    parser.add_argument('--gcn_order', type=int, default=2)
    parser.add_argument('--tanh_scale', type=float, default=40.0)
    parser.add_argument('--tau', type=float, default=5.0)
    parser.add_argument('--n_iter', type=int, default=60)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--shift', type=int, default=-1, help='Shift parameter for rolling transpose matrix (negative integer)')
    parser.add_argument('--distance_scale', type=float, default=5.0, 
                       help='Scaling factor for distance matrix in adjacency computation (exp(-dist/scale))')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Optimization improvements
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'adamw', 'radam', 'lion'],
                       help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use warmup + cosine annealing scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=15,
                       help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate for scheduler')
    
    # Stability improvements
    parser.add_argument('--early_stopping', action='store_true',
                       help='Use early stopping')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--adaptive_grad_clip', action='store_true',
                       help='Use adaptive gradient clipping')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--loss_smoothing', type=float, default=0.0,
                       help='Loss smoothing factor')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Visualize tours during validation')
    parser.add_argument('--viz_freq', type=int, default=20, help='Visualization frequency (epochs)')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Using seed: {args.seed}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize WandB
#    if args.use_wandb:
#        # Check if WandB is properly authenticated
#        try:
#            wandb.api.api_key
#        except Exception:
#            setup_wandb_auth()
#            print("\nPlease authenticate WandB first, then run the script again.")
#            return
#        
#        # Create run name if not provided
#        if args.wandb_name is None:
#            args.wandb_name = f"Tau{args.tau}_NoiseScale_{args.noise_scale}_nIter{args.n_iter}_shift{args.shift}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_dist{args.distance_scale}_seed{args.seed}"
#        
#        # Set WandB mode
#        wandb_mode = "offline" if args.wandb_offline else "online"
#        
#        wandb.init(
#            project=args.wandb_project,
#            entity=args.wandb_entity,
#            name=args.wandb_name,
#            tags=args.wandb_tags,
#            config=vars(args),
#            mode=wandb_mode
#        )
#        print(f"WandB initialized ({wandb_mode}): {wandb.run.url if not args.wandb_offline else 'offline mode'}")


    # Load datasets
    print("Loading datasets...")
    train_coords, _ = load_tsp_dataset(args.train_data)
    val_coords, _ = load_tsp_dataset(args.val_data)
    
    # Create simple datasets (just coordinates)
    print("Creating datasets with coordinates only...")
    train_dataset = SimpleTSPDataset(train_coords)
    val_dataset = SimpleTSPDataset(val_coords)
    
    # Create dataloaders
    train_loader = SimpleTSPDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = SimpleTSPDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create SCT model
    input_dim = 2  # Just x, y coordinates
    output_dim = train_coords.shape[1]  # Number of cities
    
    print(f"Creating SCT model...")
    print(f"Input dim: {input_dim} (x, y coordinates)")
    print(f"Output dim: {output_dim} (number of cities)")
    print(f"Shift parameter: {args.shift}")
    print(f"Distance scaling factor: {args.distance_scale}")
    
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

 
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Create optimizer with better defaults
    print(f"Using {args.optimizer.upper()} optimizer with weight decay {args.weight_decay}")
    optimizer = create_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = WarmupCosineScheduler(
            optimizer, args.warmup_epochs, args.epochs, args.lr, args.min_lr
        )
        print(f"Using warmup ({args.warmup_epochs}) + cosine annealing scheduler")

    # Early stopping
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.patience)
        print(f"Using early stopping with patience {args.patience}")

    # Gradient clipping
    grad_clipper = None
    if args.adaptive_grad_clip:
        grad_clipper = GradientClipper(model)
        print("Using adaptive gradient clipping")

    # Mixed precision
    if args.mixed_precision:
        print("Using mixed precision training")

    # Training loop
    best_val_length = float('inf')
    training_history = {'train_loss': [], 'train_length': [], 'val_length': [], 'lr': []}
    
    print(f"\nStarting stable training for {args.epochs} epochs...")
    print(f"Optimization features: {args.optimizer}, scheduler={args.use_scheduler}, "
          f"early_stop={args.early_stopping}, grad_clip={args.adaptive_grad_clip}, "
          f"mixed_prec={args.mixed_precision}")
    print(f"Shift parameter: {args.shift}")
    print(f"Distance scaling factor: {args.distance_scale}")
    
    for epoch in range(args.epochs):
        # Update learning rate
        current_lr = args.lr
        if scheduler:
            current_lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_length = train_epoch(
            model, train_loader, optimizer, device,
            shift=args.shift,
            distance_scale=args.distance_scale,
            grad_clipper=grad_clipper,
            use_mixed_precision=args.mixed_precision,
            loss_smoothing=args.loss_smoothing
        )
        
        # Validate
        should_visualize = args.visualize and (epoch + 1) % args.viz_freq == 0
        val_length = validate_epoch(model, val_loader, device, 
                                   shift=args.shift, 
                                   distance_scale=args.distance_scale,
                                   epoch=epoch, visualize=should_visualize)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_length'].append(train_length)
        training_history['val_length'].append(val_length)
        training_history['lr'].append(current_lr)
         # Log to WandB
#        if args.use_wandb:
#            log_dict = {
#                #'epoch': epoch + 1,
#                'train_loss': train_loss,
#                #'train_length': train_length,
#                'val_length': val_length,
#                'learning_rate': current_lr,
#                'best_val_length': best_val_length
#            }
#            
#            # Add gradient norm if using gradient clipping
#            if grad_clipper and hasattr(grad_clipper, 'grad_history') and grad_clipper.grad_history:
#                log_dict['grad_norm'] = grad_clipper.grad_history[-1]
#            
#            #wandb.log(log_dict)
#            wandb.log(log_dict, step=epoch + 1)
#        
        print(f"Epoch {epoch+1:3d}: "
              f"LR={current_lr:.2e}, Train Loss={train_loss:.4f}, "
              f"Train Length={train_length:.2f}, Val Length={val_length:.2f}")
        
        # Save best model
        if val_length < best_val_length:
            best_val_length = val_length
#            model_save_path = (f'SaveModels/2MTraining_best_stable_sct_model_'
            model_save_path = (f'SaveModels/best_stable_sct_model_'
                             f'size_{output_dim}_hidden_{args.hidden_dim}_'
                             f'{args.optimizer}_tau_{args.tau}_'
                             f'n_iter_{args.n_iter}_noise_{args.noise_scale}_'
                             f'shift_{args.shift}_dist_scale_{args.distance_scale}_'
                             f'n_layers{args.n_layers}_'
                             f'seed_{args.seed}.pt')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_length': val_length,
                'training_history': training_history,
                'args': args
            }, model_save_path)
            print(f"★ New best model! Val Length: {val_length:.2f}")
            # Log best model to WandB
            #if args.use_wandb:
            #    wandb.log({'best_val_length_updated': best_val_length})

        
        # Early stopping check
        if early_stopping and early_stopping(val_length, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best validation length: {best_val_length:.2f}")
            #if args.use_wandb:
            #    #wandb.log({'early_stopped_epoch': epoch + 1})
            #    wandb.log({'early_stopped_epoch': epoch + 1}, step=epoch + 1)
            break
        
        # Check for training instability
        if len(training_history['val_length']) > 10:
            recent_vals = training_history['val_length'][-10:]
            if max(recent_vals) - min(recent_vals) > 100:  # Large variance
                print("Warning: Training shows high variance. Consider reducing learning rate.")
                #if args.use_wandb:
                #    wandb.log({'training_unstable': True, 'val_variance': max(recent_vals) - min(recent_vals)})

    print(f"\nTraining completed!")
    print(f"Best validation length: {best_val_length:.2f}")
    print(f"Training history saved in model checkpoint")

    # Final WandB logging
#    if args.use_wandb:
#        wandb.log({
#            'final_best_val_length': best_val_length,
#            'total_epochs': len(training_history['train_loss'])
#        })
#
#        # Save model artifact to WandB
#        if os.path.exists(model_save_path):
#            artifact = wandb.Artifact(
#                name=f"best_model_{wandb.run.id}",
#                type="model",
#                description=f"Best Multi-Scale Graph TSP model with val_length={best_val_length:.2f}"
#            )
#            artifact.add_file(model_save_path)
#            wandb.log_artifact(artifact)
#
#
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs_range = range(1, len(training_history['train_loss']) + 1)
        
        # Loss curve
        ax1.plot(epochs_range, training_history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Tour length curves
        ax2.plot(epochs_range, training_history['train_length'], label='Train')
        ax2.plot(epochs_range, training_history['val_length'], label='Validation')
        ax2.set_title('Tour Lengths')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Tour Length')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs_range, training_history['lr'])
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Validation length (zoomed)
        ax4.plot(epochs_range, training_history['val_length'])
        ax4.set_title('Validation Length (Detailed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Tour Length')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_curves_shift_{args.shift}_dist_scale_{args.distance_scale}_seed_{args.seed}.png', dpi=150)
        print(f"Training curves saved to: training_curves_shift_{args.shift}_dist_scale_{args.distance_scale}_seed_{args.seed}.png")

        # Log plot to WandB
#        if args.use_wandb:
#            wandb.log({"training_curves": wandb.Image(plot_path)})

    except ImportError:
        print("Matplotlib not available, skipping training curves plot")

    # Clean up WandB
#    if args.use_wandb:
#        wandb.finish()

if __name__ == "__main__":
    main()
