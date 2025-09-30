#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCT-TSP trainer with: single-node DDP (multi-GPU), safe checkpointing, auto-resume,
Warmup+Cosine scheduler, adaptive grad clipping, early stopping, SIGTERM handler,
periodic saving by steps/minutes/epochs, milestones, and best checkpointing.

Requires project modules:
- cyclefastsags_stable.EnhancedUltraFastSCTGNN as GNN
- data_generator: SimpleTSPDataset, SimpleTSPDataLoader, load_tsp_dataset
- hardpermutation.to_exact_permutation_batched
- utsploss.tsp_permutation_loss
- (optional) tsp_visualization.visualize_validation_tours
"""

import argparse
import os
import time
import math
import signal
from typing import Optional, Dict, Any

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
import numpy as np

#from sagmodel import FastSCTGNN as GNN
#from cyclefastsags_stable_precomputemultinode import EnhancedUltraFastSCTGNN as GNN
from sagnn import EnhancedUltraFastSCTGNN as GNN
from data_generator import SimpleTSPDataset, SimpleTSPDataLoader, load_tsp_dataset
from hardpermutation import to_exact_permutation_batched
from utsploss import tsp_permutation_loss


from torch.utils.data import DistributedSampler
from data_generator_ddp import SimpleTSPDataset, load_tsp_dataset, build_dataloader

try:
    from tsp_visualization import visualize_validation_tours  # noqa: F401
    VIZ_AVAILABLE = True
except Exception:
    VIZ_AVAILABLE = False

# ----------------------------- utils ----------------------------- #

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def is_dist():
    return dist.is_initialized()


def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def atomic_save(path: str, payload: Dict[str, Any]):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if is_main_process():
        print(f"[Checkpoint] Saved: {path}")


# ------------------------- schedulers/clippers ------------------------- #

class WarmupCosineScheduler:
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




class WarmupExponentialScheduler:
    """Warmup + Exponential Decay learning rate scheduler (epoch-based)."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6, decay_rate=0.995):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Exponential decay
            steps = epoch - self.warmup_epochs
            lr = self.base_lr * (self.decay_rate ** steps)
            lr = max(lr, self.min_lr)

        # Apply to optimizer
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.001, restore_best_weights=True):
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
        stop = self.counter >= self.patience
        if stop and self.restore_best_weights and self.best_weights is not None:
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})
        return stop


class GradientClipper:
    def __init__(self, model, clip_percentile=10):
        self.model = model
        self.clip_percentile = clip_percentile
        self.hist = []

    def clip(self):
        norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                norms.append(p.grad.norm().item())
        if not norms:
            return 0.0
        cur = math.sqrt(sum(n ** 2 for n in norms))
        self.hist.append(cur)
        if len(self.hist) > 100:
            self.hist = self.hist[-100:]
        if len(self.hist) > 10:
            import numpy as _np
            thr = float(_np.percentile(self.hist, 100 - self.clip_percentile))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max(1e-6, thr))
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return cur


# ------------------------- checkpoint helpers ------------------------- #

def rng_pack():
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
    }


def _to_bytetensor(x):
    import torch
    if isinstance(x, torch.ByteTensor):
        return x
    if isinstance(x, torch.Tensor) and x.dtype == torch.uint8:
        return x
    if isinstance(x, bytes):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.uint8)
    return None



def _coerce_cpu_bytetensor(x):
    import torch as T
    # Coerce to CPU uint8 tensor
    if isinstance(x, T.Tensor):
        return x.detach().to(device="cpu", dtype=T.uint8).contiguous()
    if isinstance(x, (bytes, bytearray, memoryview)):
        return T.tensor(list(x), dtype=T.uint8)
    if isinstance(x, (list, tuple)):
        return T.tensor(x, dtype=T.uint8)
    return None

def _valid_len(x, expected):
    try:
        return x is not None and x.numel() == expected
    except Exception:
        return False

def rng_restore(state, fallback_seed: int | None = None, rank: int = 0):
    import torch, numpy as np

    if not state:
        return

    # ---- PyTorch CPU RNG ----
    try:
        exp_len = torch.random.get_rng_state().numel()  # expected size on this build
        t = state.get("torch")
        bt = _coerce_cpu_bytetensor(t)
        if _valid_len(bt, exp_len):
            torch.set_rng_state(bt)
        else:
            print(f"[Resume] Skip torch RNG: invalid/unknown format or length "
                  f"(got {None if bt is None else bt.numel()}, expect {exp_len})")
            if fallback_seed is not None:
                torch.manual_seed(fallback_seed + int(rank))
    except Exception as e:
        print(f"[Resume] Skip torch RNG (error): {e}")
        if fallback_seed is not None:
            torch.manual_seed(fallback_seed + int(rank))

    # ---- CUDA RNG (optional) ----
    try:
        c = state.get("cuda")
        if c is not None and torch.cuda.is_available():
            exp_cuda = torch.cuda.get_rng_state(0).numel()
            bt_list = []
            if isinstance(c, (list, tuple)):
                for i, e in enumerate(c):
                    bt_i = _coerce_cpu_bytetensor(e)
                    if _valid_len(bt_i, exp_cuda):
                        bt_list.append(bt_i)
                    else:
                        print(f"[Resume] Skip CUDA RNG for device idx {i}: bad length "
                              f"(got {None if bt_i is None else bt_i.numel()}, expect {exp_cuda})")
                if bt_list:
                    torch.cuda.set_rng_state_all(bt_list)
            else:
                print("[Resume] Skip CUDA RNG: unsupported format")
    except Exception as e:
        print(f"[Resume] Skip CUDA RNG (error): {e}")

    # ---- NumPy RNG ----
    try:
        n = state.get("numpy")
        if n is not None:
            if isinstance(n, dict) and {"bitgen","state","pos","has_gauss","cached_gaussian"} <= set(n.keys()):
                np_state = (
                    n["bitgen"],
                    np.array(n["state"], dtype=np.uint32),
                    int(n["pos"]),
                    bool(n["has_gauss"]),
                    float(n["cached_gaussian"]),
                )
                np.random.set_state(np_state)
            else:
                # legacy tuple path if still present
                np.random.set_state(n)
    except Exception as e:
        print(f"[Resume] Skip NumPy RNG (error): {e}")


#def rng_restore(state):
#    import torch, numpy as np
#    if not state:
#        return
#
#    t = state.get('torch')
#    if t is not None:
#        bt = _to_bytetensor(t)
#        if bt is not None:
#            torch.set_rng_state(bt)
#        else:
#            print("[Resume] Skip torch RNG: unsupported format")
#
#    c = state.get('cuda')
#    if c is not None and torch.cuda.is_available():
#        if isinstance(c, (list, tuple)):
#            bt_list = []
#            for e in c:
#                bt = _to_bytetensor(e)
#                if bt is not None:
#                    bt_list.append(bt)
#            if bt_list:
#                torch.cuda.set_rng_state_all(bt_list)
#        else:
#            print("[Resume] Skip CUDA RNG: unsupported format")
#
#    n = state.get('numpy')
#    if n is not None:
#        try:
#            if isinstance(n, dict) and {'bitgen','state','pos','has_gauss','cached_gaussian'} <= set(n.keys()):
#                np_state = (
#                    n['bitgen'],
#                    np.array(n['state'], dtype=np.uint32),
#                    int(n['pos']),
#                    bool(n['has_gauss']),
#                    float(n['cached_gaussian']),
#                )
#                np.random.set_state(np_state)
#            else:
#                np.random.set_state(n)  # legacy tuple path
#        except Exception:
#            print("[Resume] Skip NumPy RNG: unsupported format")
#

#def rng_restore(state):
#    if not state:
#        return
#    if state.get('torch') is not None:
#        torch.set_rng_state(state['torch'])
#    if state.get('cuda') is not None and torch.cuda.is_available():
#        torch.cuda.set_rng_state_all(state['cuda'])
#    if state.get('numpy') is not None:
#        np.random.set_state(state['numpy'])


def pack_ckpt(args, model, optimizer, epoch, global_step, best_val, history):
    return {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'model_state_dict': (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_length': float(best_val),
        'history': history,
        'args': vars(args),
        'rng_state': rng_pack(),
        'save_time': time.time(),
    }


#def load_ckpt(path: str, map_location: str):
#    return torch.load(path, map_location=map_location)

def load_ckpt(path: str, map_location: str):
    import torch
    from torch.serialization import add_safe_globals

    # Allowlist legacy NumPy globals found in older ckpts
    try:
        import numpy as _np
        import numpy.core.multiarray as _np_ma
        # add ndarray & dtype classes + reconstruct + the specific UInt32 dtype class
        add_safe_globals([
            _np_ma._reconstruct,
            _np.ndarray,
            _np.dtype,
            _np.dtypes.UInt32DType,         # <-- this was missing in your run
        ])
    except Exception:
        pass

    # Try safe first
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e1:
        print(f"[Resume] weights_only=True failed: {e1}")

    # If you TRUST the checkpoint (your own file), allow unsafe path
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e2:
        print(f"[Resume] weights_only=False also failed: {e2}")
        raise

# ------------------------- train / validate ------------------------- #

def train_epoch(model, loader, optimizer, device, gstate: dict, shift=-1, distance_scale=5.0,
                grad_clipper: Optional[GradientClipper] = None, mixed=False,
                loss_smoothing=0.0, saver=None):
    model.train();  setattr(model, 'inference', False)
    tot_loss = 0.0; tot_len = 0.0; nb = 0
    #scaler = torch.cuda.amp.GradScaler(enabled=mixed)
    scaler = torch.amp.GradScaler('cuda', enabled=mixed)
    smooth = None

    for coords in tqdm(loader, desc=f"Train {gstate['epoch']}", disable=not is_main_process()):
        coords = coords.to(device)
        dist_mat = torch.cdist(coords, coords)
        adj = torch.exp(-dist_mat / distance_scale)
        node_feats = coords

        optimizer.zero_grad(set_to_none=True)
        if mixed:
            with torch.cuda.amp.autocast():
                P, logits = model(node_feats, adj)
                tour_lens, _ = tsp_permutation_loss(P, dist_mat, shift)
                loss = tour_lens.mean()
            scaler.scale(loss).backward()
            if grad_clipper: scaler.unscale_(optimizer); grad_clipper.clip()
            scaler.step(optimizer); scaler.update()
        else:
            P, logits = model(node_feats, adj)
            tour_lens, _ = tsp_permutation_loss(P, dist_mat, shift)
            loss = tour_lens.mean(); loss.backward()
            if grad_clipper: grad_clipper.clip()
            optimizer.step()

        # smoothing (log only)
        smooth = loss.item() if smooth is None else (loss_smoothing * loss.item() + (1.0 - loss_smoothing) * smooth)
        tot_loss += loss.item(); tot_len += tour_lens.mean().item(); nb += 1

        gstate['global_step'] += 1
        if saver is not None: saver.maybe_save(gstate)

        if not math.isfinite(loss.item()) or loss.item() > 1e6:
            if is_main_process(): print("[Warn] Non-finite loss; break epoch.")
            break

    return tot_loss / max(1, nb), tot_len / max(1, nb)


essential_viz_note = "(Visualization disabled)" if not VIZ_AVAILABLE else ""

def validate_epoch(model, loader, device, shift=-1, distance_scale=5.0, epoch=None, visualize=False):
    model.eval(); setattr(model, 'inference', True)
    tot = 0.0; nb = 0
    first_coords = None; first_heat = None
    with torch.no_grad():
        for bidx, coords in enumerate(tqdm(loader, desc=f"Valid {epoch}", disable=not is_main_process())):
            coords = coords.to(device)
            dist_mat = torch.cdist(coords, coords)
            adj = torch.exp(-dist_mat / distance_scale)
            node_feats = coords

            logits = model(node_feats, adj)
            P_hard = to_exact_permutation_batched(logits)
            vlen, heat = tsp_permutation_loss(P_hard, dist_mat, shift)
            tot += vlen.mean().item(); nb += 1

            if bidx == 0 and visualize and VIZ_AVAILABLE and epoch is not None:
                first_coords = coords.detach().cpu(); first_heat = heat.detach().cpu()

    if visualize and VIZ_AVAILABLE and first_coords is not None:
        try:
            visualize_validation_tours(first_coords, first_heat, epoch)
        except Exception as e:
            if is_main_process(): print(f"[Viz] Failed: {e}")
    return tot / max(1, nb)


# ------------------------- periodic saver ------------------------- #

class PeriodicSaver:
    def __init__(self, args, model, optimizer, save_dir: str):
        self.args = args; self.model = model; self.optimizer = optimizer
        self.save_dir = ensure_dir(save_dir); self.last_time = time.time()

    def p(self, name: str):
        return os.path.join(self.save_dir, name)

    def save_last(self, epoch: int, gstep: int, best_val: float, history: dict):
        payload = pack_ckpt(self.args, self.model, self.optimizer, epoch, gstep, best_val, history)
        atomic_save(self.p('last.ckpt'), payload)

    def maybe_save(self, gstate: dict):
        steps = int(self.args.save_every_steps)
        if steps > 0 and gstate['global_step'] % steps == 0 and is_main_process():
            self.save_last(gstate['epoch'], gstate['global_step'], gstate['best_val'], gstate['history']); self.last_time = time.time(); return
        minutes = float(self.args.save_every_minutes)
        if minutes > 0 and (time.time() - self.last_time) >= minutes * 60 and is_main_process():
            self.save_last(gstate['epoch'], gstate['global_step'], gstate['best_val'], gstate['history']); self.last_time = time.time()


# ------------------------------ main ------------------------------ #

def create_optimizer(model, optimizer_type='adam', lr=2e-3, weight_decay=1e-4):
    opt = optimizer_type.lower()
    if opt == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == 'radam':
        try:
            from torch.optim import RAdam
            return RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        except Exception:
            print("[Info] RAdam unavailable, fallback to Adam.")
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay * 10)
        except Exception:
            print("[Info] Lion unavailable, fallback to Adam.")
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # default
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def main():
    ap = argparse.ArgumentParser(description='SCT-TSP DDP trainer with safe checkpointing and resume')

    # Reproducibility
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=8,help='DataLoader workers per process (rank).')

    # Data
    ap.add_argument('--train_data', type=str, default='data/tsp_100_uniform_train.pt')
    ap.add_argument('--val_data', type=str, default='data/tsp_100_uniform_val.pt')

    # Model
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--n_layers', type=int, default=16)
    ap.add_argument('--sct_order', type=int, default=4)
    ap.add_argument('--gcn_order', type=int, default=2)
    ap.add_argument('--tanh_scale', type=float, default=0.0)
    ap.add_argument('--tau', type=float, default=3.0)
    ap.add_argument('--n_iter', type=int, default=100)
    ap.add_argument('--noise_scale', type=float, default=0.1)
    ap.add_argument('--shift', type=int, default=-1)
    ap.add_argument('--distance_scale', type=float, default=5.0)

    # Train
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--device', type=str, default='cuda')

    # Opt & sched
    ap.add_argument('--optimizer', type=str, default='adam', choices=['adam','adamw','radam','lion'])
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--use_scheduler', action='store_true')
    ap.add_argument('--warmup_epochs', type=int, default=15)
    ap.add_argument('--min_lr', type=float, default=1e-6)

    # Stability
    ap.add_argument('--early_stopping', action='store_true')
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--adaptive_grad_clip', action='store_true')
    ap.add_argument('--mixed_precision', action='store_true')
    ap.add_argument('--loss_smoothing', type=float, default=0.0)

    # Saving & resume
    ap.add_argument('--save_dir', type=str, default='SaveModels')
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--auto_resume', action='store_true')
    ap.add_argument('--save_every_steps', type=int, default=0)
    ap.add_argument('--save_every_minutes', type=float, default=0.0)
    ap.add_argument('--save_every_epochs', type=int, default=1)
    ap.add_argument('--milestone_interval', type=int, default=50)

    # DDP
    ap.add_argument('--ddp', action='store_true', help='Enable single-node DDP; launch with torchrun')

    args = ap.parse_args()
    nw = getattr(args, 'num_workers', 8)

    # Distributed init
    if args.ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)
    save_dir = ensure_dir(args.save_dir)

    # Data
    if is_main_process(): print("[Info] Loading datasets...")
    train_coords, _ = load_tsp_dataset(args.train_data)
    val_coords, _ = load_tsp_dataset(args.val_data)
    train_set = SimpleTSPDataset(train_coords)
    val_set = SimpleTSPDataset(val_coords)

    if args.ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler   = DistributedSampler(val_set, shuffle=False)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,                 # sampler controls order
            sampler=train_sampler,            # DistributedSampler(train_set, shuffle=True)
            num_workers=nw,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,              # DistributedSampler(val_set, shuffle=False) or None
            num_workers=nw,
            pin_memory=True,
        )

    else:
    # Non-DDP path: keep your simple loader
        train_loader = SimpleTSPDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader   = SimpleTSPDataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    # Model / Opt / Sched
    out_dim = train_coords.shape[1]
#    model = GNN(
#        input_dim=2, hidden_dim=args.hidden_dim, output_dim=out_dim,
#        n_layers=args.n_layers, sct_order=args.sct_order, gcn_order=args.gcn_order,
#        tanh_scale=args.tanh_scale, tau=args.tau, n_iter=args.n_iter,
#        noise_scale=args.noise_scale, inference_mode=False,netdropout=args.dropout
#    ).to(device)
    model = GNN(
        input_dim=2, hidden_dim=args.hidden_dim, output_dim=out_dim,
        n_layers=args.n_layers, sct_order=args.sct_order, gcn_order=args.gcn_order,
        tanh_scale=args.tanh_scale, tau=args.tau, n_iter=args.n_iter,
        noise_scale=args.noise_scale, inference_mode=False,ff_dropout=args.dropout
    ).to(device)
    if args.ddp:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,     # <- Only use find_unused_parameters=True when your model has conditional execution paths 
            static_graph=False               # (optional) keep False since the graph differs train vs. eval
        )
#        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    optimizer = create_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    scheduler = None
    if args.use_scheduler:
        scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.lr, args.min_lr)

    early = EarlyStopping(patience=args.patience) if args.early_stopping else None
    gclip = GradientClipper(model) if args.adaptive_grad_clip else None

    # State
    start_epoch = 0
    best_val = float('inf')
    history = {'train_loss': [], 'train_length': [], 'val_length': [], 'lr': []}
    gstate = {'epoch': 0, 'global_step': 0, 'best_val': best_val, 'history': history}

    # Resume
    last_ckpt = os.path.join(save_dir, 'last.ckpt')
    ckpt_path = ''
    if args.resume:
        ckpt_path = args.resume
    elif args.auto_resume and os.path.isfile(last_ckpt):
        ckpt_path = last_ckpt

    if ckpt_path:
        if is_main_process(): print(f"[Resume] Loading from {ckpt_path}")
        ckpt = load_ckpt(ckpt_path, map_location=str(device))
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt['model_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        except Exception as e:
            if is_main_process(): print(f"[Resume] Optimizer state not loaded: {e}")
        rank = dist.get_rank() if is_dist() else 0
        rng_restore(ckpt.get("rng_state"), fallback_seed=args.seed, rank=rank)

        #rng_restore(ckpt.get('rng_state'))
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        gstate['global_step'] = int(ckpt.get('global_step', 0))
        history = ckpt.get('history', history)
        best_val = float(ckpt.get('best_val_length', float('inf')))
        gstate['history'] = history
        gstate['best_val'] = best_val
        if is_main_process():
            print(f"[Resume] start_epoch={start_epoch}, global_step={gstate['global_step']}, best_val={best_val:.2f}")
    barrier()

    # Saver & SIGTERM
    saver = PeriodicSaver(args, model, optimizer, save_dir)

    def _on_term(sig, frame):
        if is_main_process():
            print(f"[Signal] Caught {sig}. Saving last.ckpt and exiting...")
            saver.save_last(gstate['epoch'], gstate['global_step'], gstate['best_val'], gstate['history'])
        barrier(); os._exit(0)

    signal.signal(signal.SIGTERM, _on_term)

    if is_main_process():
        n_params = sum(p.numel() for p in (model.module if isinstance(model, DDP) else model).parameters() if p.requires_grad)
        print(f"[Info] Params: {n_params:,}")
        print(f"[Info] Starting from epoch {start_epoch} to {args.epochs}")

    best_payload = None

    for epoch in range(start_epoch, args.epochs):
        gstate['epoch'] = epoch
        if args.ddp and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        cur_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            cur_lr = scheduler.step(epoch)

        tr_loss, tr_len = train_epoch(
            model, train_loader, optimizer, device, gstate,
            shift=args.shift, distance_scale=args.distance_scale,
            grad_clipper=gclip, mixed=args.mixed_precision,
            loss_smoothing=args.loss_smoothing, saver=saver
        )

        val_len = validate_epoch(
            model, val_loader, device,
            shift=args.shift, distance_scale=args.distance_scale,
            epoch=epoch, visualize=False
        )

        if is_main_process():
            history['train_loss'].append(tr_loss)
            history['train_length'].append(tr_len)
            history['val_length'].append(val_len)
            history['lr'].append(cur_lr)
            print(f"Epoch {epoch+1:4d} | LR={cur_lr:.2e} | TrainLoss={tr_loss:.4f} | TrainLen={tr_len:.2f} | ValLen={val_len:.2f}")

            # save last every N epochs
            if (epoch + 1) % max(1, args.save_every_epochs) == 0:
                saver.save_last(epoch, gstate['global_step'], best_val, history)

            # best & milestone
            if val_len < best_val:
                best_val = val_len; gstate['best_val'] = best_val
                best_payload = pack_ckpt(args, model, optimizer, epoch, gstate['global_step'], best_val, history)
                bestmodelname=(
                    f"ddpbest_"
                    f"size_{out_dim}_hidden_{args.hidden_dim}_"
                    f"{args.optimizer}_tau_{args.tau}_"
                    f"n_iter_{args.n_iter}_noise_{args.noise_scale}_"
                    f"shift_{args.shift}_dist_scale_{args.distance_scale}_"
                    f"n_layers{args.n_layers}_seed_{args.seed}_dropout{args.dropout}.ckpt"
                )
                atomic_save(os.path.join(save_dir, bestmodelname), best_payload)
#                atomic_save(os.path.join(save_dir, 'best.ckpt'), best_payload)
                print(f"★ New best validation length: {val_len:.2f}")

            if (epoch + 1) % max(1, args.milestone_interval) == 0 and best_payload is not None:
                mname = (
                    f"ddpmilestone_best_up_to_epoch_{epoch+1}_"
                    f"size_{out_dim}_hidden_{args.hidden_dim}_"
                    f"{args.optimizer}_tau_{args.tau}_"
                    f"n_iter_{args.n_iter}_noise_{args.noise_scale}_"
                    f"shift_{args.shift}_dist_scale_{args.distance_scale}_"
                    f"n_layers{args.n_layers}_seed_{args.seed}_dropout{args.dropout}.ckpt"
                )
                atomic_save(os.path.join(save_dir, mname), best_payload)

            # instability warning
            if len(history['val_length']) > 10:
                recent = history['val_length'][-10:]
                if max(recent) - min(recent) > 100.0:
                    print("[Warn] High val variance; consider smaller LR or enable grad clip.")

        barrier()

        # early stop (checked on main only, but we broadcast stop with barrier by simply breaking after barrier)
        if early and is_main_process() and early(val_len, (model.module if isinstance(model, DDP) else model)):
            print(f"[EarlyStop] at epoch {epoch+1}, best={best_val:.2f}")
            saver.save_last(epoch, gstate['global_step'], best_val, history)
            # tell others to stop by writing a file flag
            with open(os.path.join(save_dir, '.early_stop'), 'w') as f:
                f.write('1')
        barrier()
        if os.path.isfile(os.path.join(save_dir, '.early_stop')):
            break

    if is_main_process():
        print("Training completed.")
        print(f"Best validation length: {best_val:.2f}")


if __name__ == '__main__':
    main()

