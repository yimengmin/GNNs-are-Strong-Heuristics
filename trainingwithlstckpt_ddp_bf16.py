#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCT-TSP trainer (rewritten):
- Single-node DDP (torchrun) or single-GPU
- Warmup + Cosine LR
- AMP (bf16/fp16) with correct precision fences
- Adaptive grad clipping
- Early stopping
- Safe checkpointing + auto-resume
- Periodic saving by steps/minutes/epochs + milestone & best checkpoint

Project deps expected on PYTHONPATH:
  - sagnn.EnhancedUltraFastSCTGNN as GNN
  - data_generator.SimpleTSPDataset, SimpleTSPDataLoader, load_tsp_dataset
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

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# ---------- project modules ----------
from sagnn_bf16 import EnhancedUltraFastSCTGNN as GNN
from data_generator import SimpleTSPDataset, SimpleTSPDataLoader, load_tsp_dataset
from hardpermutation import to_exact_permutation_batched
from utsploss import tsp_permutation_loss

try:
    from tsp_visualization import visualize_validation_tours  # noqa: F401
    VIZ_AVAILABLE = True
except Exception:
    VIZ_AVAILABLE = False


# ========================= utils =========================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def is_dist() -> bool:
    return dist.is_initialized()


def is_main() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def atomic_save(path: str, payload: Dict[str, Any]):
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if is_main():
        print(f"[Checkpoint] Saved: {path}")


# =================== schedulers & helpers ===================

class WarmupCosineScheduler:
    """Epoch-based linear warmup then cosine to min_lr."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.warm = warmup_epochs
        self.maxe = max_epochs
        self.base = base_lr
        self.min = min_lr

    def step(self, epoch: int) -> float:
        if epoch < self.warm:
            lr = self.base * (epoch + 1) / max(1, self.warm)
        else:
            total = max(1, self.maxe - self.warm)
            prog = (epoch - self.warm) / total
            prog = min(max(prog, 0.0), 1.0)
            lr = self.min + (self.base - self.min) * 0.5 * (1 + math.cos(math.pi * prog))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-3, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best = float("inf")
        self.count = 0
        self.best_state = None

    def __call__(self, val_value: float, model) -> bool:
        improved = val_value < self.best - self.min_delta
        if improved:
            self.best = val_value
            self.count = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.count += 1

        stop = self.count >= self.patience
        if stop and self.restore_best and self.best_state is not None:
            dev = next(model.parameters()).device
            model.load_state_dict({k: v.to(dev) for k, v in self.best_state.items()})
        return stop


class GradientClipper:
    """Adaptive clip to a percentile of recent global grad norms."""
    def __init__(self, model, clip_percentile=10):
        self.model = model
        self.p = clip_percentile
        self.hist = []

    def clip(self) -> float:
        norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                norms.append(p.grad.norm().item())
        if not norms:
            return 0.0
        cur = math.sqrt(sum(n * n for n in norms))
        self.hist.append(cur)
        if len(self.hist) > 100:
            self.hist = self.hist[-100:]
        if len(self.hist) > 10:
            import numpy as _np
            thr = float(_np.percentile(self.hist, 100 - self.p))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max(1e-6, thr))
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        return cur


# =================== RNG pack / resume helpers ===================

def rng_pack():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
    }


def _coerce_cpu_bytetensor(x):
    T = torch
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
    if not state:
        return
    # torch CPU
    try:
        exp = torch.random.get_rng_state().numel()
        bt = _coerce_cpu_bytetensor(state.get("torch"))
        if _valid_len(bt, exp):
            torch.set_rng_state(bt)
        elif fallback_seed is not None:
            torch.manual_seed(fallback_seed + int(rank))
    except Exception:
        if fallback_seed is not None:
            torch.manual_seed(fallback_seed + int(rank))
    # CUDA
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
                if bt_list:
                    torch.cuda.set_rng_state_all(bt_list)
    except Exception:
        pass
    # numpy
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
                np.random.set_state(n)
    except Exception:
        pass


def pack_ckpt(args, model, optimizer, epoch, gstep, best_val, history):
    return {
        "epoch": int(epoch),
        "global_step": int(gstep),
        "model_state_dict": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_length": float(best_val),
        "history": history,
        "args": vars(args),
        "rng_state": rng_pack(),
        "save_time": time.time(),
    }


def load_ckpt(path: str, map_location: str):
    import torch
    from torch.serialization import add_safe_globals
    try:
        import numpy as _np
        import numpy.core.multiarray as _np_ma
        add_safe_globals([
            _np_ma._reconstruct,
            _np.ndarray,
            _np.dtype,
            _np.dtypes.UInt32DType,
        ])
    except Exception:
        pass
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


# =================== training / validation ===================

def train_epoch(
    model, loader, optimizer, device, gstate: dict,
    shift=-1, distance_scale=5.0,
    grad_clipper: Optional[GradientClipper] = None,
    mixed=False, amp_dtype="bf16",
    loss_smoothing=0.0, saver=None
):
    model.train();  setattr(model, "inference", False)
    tot_loss = 0.0; tot_len = 0.0; nb = 0

    use_scaler = bool(mixed and amp_dtype == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    for coords in tqdm(loader, desc=f"Train {gstate['epoch']}", disable=not is_main()):
        coords = coords.to(device)
        dist_mat = torch.cdist(coords, coords)                     # [B, N, N] float32
        adj = torch.exp(-dist_mat / distance_scale)
        node_feats = coords                                        # [B, N, 2]

        optimizer.zero_grad(set_to_none=True)
        if mixed:
            dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                P, _Z = model(node_feats, adj)                     # model may autocast internally too
                tour_lens, _ = tsp_permutation_loss(P, dist_mat, shift)
                loss = tour_lens.mean()
            if use_scaler:
                scaler.scale(loss).backward()
                if grad_clipper:
                    scaler.unscale_(optimizer); grad_clipper.clip()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if grad_clipper: grad_clipper.clip()
                optimizer.step()
        else:
            P, _Z = model(node_feats, adj)
            tour_lens, _ = tsp_permutation_loss(P, dist_mat, shift)
            loss = tour_lens.mean()
            loss.backward()
            if grad_clipper: grad_clipper.clip()
            optimizer.step()

        tot_loss += float(loss.item())
        tot_len  += float(tour_lens.mean().item())
        nb += 1

        gstate["global_step"] += 1
        if saver is not None:
            saver.maybe_save(gstate)

        if not math.isfinite(loss.item()) or loss.item() > 1e6:
            if is_main(): print("[Warn] Non-finite loss; breaking epoch.")
            break

    return tot_loss / max(1, nb), tot_len / max(1, nb)


def validate_epoch(
    model, loader, device, shift=-1, distance_scale=5.0,
    epoch=None, visualize=False, mixed=False, amp_dtype="bf16"
):
    """Important: we run the *permutation* + *tsp loss* in full precision (fp32),
    exactly as you requested, even when the forward pass uses AMP.
    """
    model.eval(); setattr(model, "inference", True)
    tot = 0.0; nb = 0

    first_coords = None; first_heat = None

    with torch.no_grad():
        for bidx, coords in enumerate(tqdm(loader, desc=f"Valid {epoch}", disable=not is_main())):
            coords = coords.to(device)
            dist_mat = torch.cdist(coords, coords)                  # fp32
            adj = torch.exp(-dist_mat / distance_scale)
            node_feats = coords

            # Forward may be autocast…
            if mixed:
                dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
                with torch.amp.autocast("cuda", dtype=dtype):
                    logits = model(node_feats, adj)
            else:
                logits = model(node_feats, adj)

            # …but the following MUST be the original precision (fp32):
            # Disable autocast and cast inputs as needed.
            with torch.amp.autocast("cuda", enabled=False):
                logits_f32 = logits.float()
                P_hard = to_exact_permutation_batched(logits_f32)      # << original precision
                vlen, heat = tsp_permutation_loss(P_hard, dist_mat, shift)  # << original precision

            tot += float(vlen.mean().item()); nb += 1

            if bidx == 0 and visualize and VIZ_AVAILABLE and epoch is not None:
                first_coords = coords.detach().cpu(); first_heat = heat.detach().cpu()

    if visualize and VIZ_AVAILABLE and first_coords is not None:
        try:
            visualize_validation_tours(first_coords, first_heat, epoch)
        except Exception as e:
            if is_main(): print(f"[Viz] Failed: {e}")

    return tot / max(1, nb)


# =================== periodic saver ===================

class PeriodicSaver:
    def __init__(self, args, model, optimizer, save_dir: str):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.save_dir = ensure_dir(save_dir)
        self.last_time = time.time()

    def p(self, name: str) -> str:
        return os.path.join(self.save_dir, name)

    def save_last(self, epoch: int, gstep: int, best_val: float, history: dict):
        payload = pack_ckpt(self.args, self.model, self.optimizer, epoch, gstep, best_val, history)
        atomic_save(self.p("last.ckpt"), payload)

    def maybe_save(self, gstate: dict):
        steps = int(self.args.save_every_steps)
        if steps > 0 and gstate["global_step"] % steps == 0 and is_main():
            self.save_last(gstate["epoch"], gstate["global_step"], gstate["best_val"], gstate["history"])
            self.last_time = time.time()
            return
        minutes = float(self.args.save_every_minutes)
        if minutes > 0 and (time.time() - self.last_time) >= minutes * 60 and is_main():
            self.save_last(gstate["epoch"], gstate["global_step"], gstate["best_val"], gstate["history"])
            self.last_time = time.time()


# =================== optimizer factory ===================

def create_optimizer(model, optimizer_type="adam", lr=2e-3, weight_decay=1e-4):
    opt = optimizer_type.lower()
    if opt == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == "radam":
        try:
            from torch.optim import RAdam
            return RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        except Exception:
            if is_main(): print("[Info] RAdam unavailable, fallback to Adam.")
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay * 10)
        except Exception:
            if is_main(): print("[Info] Lion unavailable, fallback to Adam.")
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # default
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# =================== main ===================

def main():
    ap = argparse.ArgumentParser("SCT-TSP Trainer (AMP-ready)")

    # Repro
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader workers per process (rank).")

    # Data
    ap.add_argument("--train_data", type=str, default="data/tsp_100_uniform_train.pt")
    ap.add_argument("--val_data", type=str,   default="data/tsp_100_uniform_val.pt")

    # Model
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--sct_order", type=int, default=4)
    ap.add_argument("--gcn_order", type=int, default=2)
    ap.add_argument("--tanh_scale", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--n_iter", type=int, default=100)
    ap.add_argument("--noise_scale", type=float, default=0.1)
    ap.add_argument("--shift", type=int, default=-1)
    ap.add_argument("--distance_scale", type=float, default=5.0)

    # Train
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", type=str, default="cuda")

    # Opt & sched
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam","adamw","radam","lion"])
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_scheduler", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=15)
    ap.add_argument("--min_lr", type=float, default=3e-4)

    # Stability
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--adaptive_grad_clip", action="store_true")
    ap.add_argument("--loss_smoothing", type=float, default=0.0)

    # AMP
    ap.add_argument("--mixed_precision", action="store_true", help="Enable AMP autocast.")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16","fp16"], help="AMP dtype (bf16 recommended).")

    # Saving & resume
    ap.add_argument("--save_dir", type=str, default="SaveModels")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--auto_resume", action="store_true")
    ap.add_argument("--save_every_steps", type=int, default=0)
    ap.add_argument("--save_every_minutes", type=float, default=0.0)
    ap.add_argument("--save_every_epochs", type=int, default=1)
    ap.add_argument("--milestone_interval", type=int, default=50)

    # DDP
    ap.add_argument("--ddp", action="store_true", help="Enable single-node DDP; launch with torchrun")

    args = ap.parse_args()
    nw = args.num_workers

    # Distributed init
    if args.ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Optional: speed up FP32 GEMMs and let AMP/BF16 cover the rest
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    save_dir = ensure_dir(args.save_dir)

    # Data
    ### not use shard memory
    if is_main(): print("[Info] Loading datasets…")
    train_coords, _ = load_tsp_dataset(args.train_data)
    val_coords, _ = load_tsp_dataset(args.val_data)
    train_set = SimpleTSPDataset(train_coords)
    val_set = SimpleTSPDataset(val_coords)
    ##################
    ## use shard memory, run shard_tsp_dataset.py first 
    # Data
    #if is_main():
    #    print("[Info] Loading datasets…")

    ## --- TRAIN DATA: per-rank shard ---
    #if args.ddp:
    #    rank = dist.get_rank()
    #    world_size = dist.get_world_size()

    #    base, ext = os.path.splitext(args.train_data)
    #    # Expect files like: tsp_500_uniform_train_6m_bf16_rank0of12.pt, ..., rank11of12.pt
    #    shard_path = f"{base}_rank{rank}of{world_size}{ext}"
    #    if is_main():
    #        print(f"[Info] Using sharded train file for rank {rank}: {shard_path}")
    #    train_coords, _ = load_tsp_dataset(shard_path)
    #else:
    #    train_coords, _ = load_tsp_dataset(args.train_data)

    ## --- VAL DATA: keep as single file for now (all ranks read same val set) ---
    #val_coords, _ = load_tsp_dataset(args.val_data)

    #train_set = SimpleTSPDataset(train_coords)
    #val_set   = SimpleTSPDataset(val_coords)


    if args.ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler   = DistributedSampler(val_set, shuffle=False)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=False,
            sampler=train_sampler, num_workers=nw, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            sampler=val_sampler, num_workers=nw, pin_memory=True
        )
    else:
        train_loader = SimpleTSPDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader   = SimpleTSPDataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    # Model
    out_dim = train_coords.shape[1]
    model = GNN(
        input_dim=2, hidden_dim=args.hidden_dim, output_dim=out_dim,
        n_layers=args.n_layers, sct_order=args.sct_order, gcn_order=args.gcn_order,
        tanh_scale=args.tanh_scale, tau=args.tau, n_iter=args.n_iter,
        noise_scale=args.noise_scale, inference_mode=False,
        ff_dropout=args.dropout,
        amp_autocast=args.mixed_precision   # let the model also wrap forward with autocast
    ).to(device)

    if args.ddp:
        model = DDP(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=False, static_graph=False
        )

    # Opt & sched
    optimizer = create_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs, args.lr, args.min_lr) \
                if args.use_scheduler else None
    early = EarlyStopping(patience=args.patience) if args.early_stopping else None
    gclip = GradientClipper(model) if args.adaptive_grad_clip else None

    # State
    start_epoch = 0
    best_val = float("inf")
    history = {"train_loss": [], "train_length": [], "val_length": [], "lr": []}
    gstate = {"epoch": 0, "global_step": 0, "best_val": best_val, "history": history}

    # Resume
    last_ckpt = os.path.join(save_dir, "last.ckpt")
    ckpt_path = args.resume if args.resume else (last_ckpt if args.auto_resume and os.path.isfile(last_ckpt) else "")
    if ckpt_path:
        if is_main(): print(f"[Resume] Loading from {ckpt_path}")
        ckpt = load_ckpt(ckpt_path, map_location=str(device))
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # move optimizer states to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        except Exception as e:
            if is_main(): print(f"[Resume] Optimizer state not loaded: {e}")
        rank = dist.get_rank() if is_dist() else 0
        rng_restore(ckpt.get("rng_state"), fallback_seed=args.seed, rank=rank)

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        gstate["global_step"] = int(ckpt.get("global_step", 0))
        history = ckpt.get("history", history)
        best_val = float(ckpt.get("best_val_length", float("inf")))
        gstate["history"] = history
        gstate["best_val"] = best_val
        if is_main():
            print(f"[Resume] start_epoch={start_epoch}, global_step={gstate['global_step']}, best_val={best_val:.2f}")
    barrier()

    # Saver & signal
    saver = PeriodicSaver(args, model, optimizer, save_dir)

    def _on_term(sig, frame):
        if is_main():
            print(f"[Signal] Caught {sig}. Saving last.ckpt and exiting…")
            saver.save_last(gstate["epoch"], gstate["global_step"], gstate["best_val"], gstate["history"])
        barrier(); os._exit(0)

    signal.signal(signal.SIGTERM, _on_term)

    if is_main():
        n_params = sum(p.numel() for p in (model.module if isinstance(model, DDP) else model).parameters() if p.requires_grad)
        print(f"[Info] Params: {n_params:,}")
        print(f"[Info] Starting from epoch {start_epoch} to {args.epochs}")
        if args.mixed_precision:
            print(f"[Info] AMP enabled (dtype={args.amp_dtype})")

    best_payload = None

    best_path = os.path.join(save_dir, "best.ckpt")
    if os.path.isfile(best_path):
        try:
            best_payload = load_ckpt(best_path, map_location=str(device))
            best_val = float(best_payload.get("best_val_length", best_val))
            gstate["best_val"] = best_val
            if is_main():
                print(f"[Init] Loaded best_payload from {best_path}, best_val={best_val:.2f}")
        except Exception as e:
            if is_main():
                print(f"[Init] Failed to load best.ckpt: {e}")


    for epoch in range(start_epoch, args.epochs):
        gstate["epoch"] = epoch
        if args.ddp:
            # refresh sampler epoch for proper shuffling
            for dl in (train_loader,):
                if isinstance(dl.sampler, DistributedSampler):
                    dl.sampler.set_epoch(epoch)

        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            cur_lr = scheduler.step(epoch)

        tr_loss, tr_len = train_epoch(
            model, train_loader, optimizer, device, gstate,
            shift=args.shift, distance_scale=args.distance_scale,
            grad_clipper=gclip, mixed=args.mixed_precision, amp_dtype=args.amp_dtype,
            loss_smoothing=args.loss_smoothing, saver=saver
        )

        val_len = validate_epoch(
            model, val_loader, device,
            shift=args.shift, distance_scale=args.distance_scale,
            epoch=epoch, visualize=False,
            mixed=args.mixed_precision, amp_dtype=args.amp_dtype
        )

        if is_main():
            history["train_loss"].append(tr_loss)
            history["train_length"].append(tr_len)
            history["val_length"].append(val_len)
            history["lr"].append(cur_lr)
            print(f"Epoch {epoch+1:4d} | LR={cur_lr:.2e} | TrainLoss={tr_loss:.4f} | TrainLen={tr_len:.2f} | ValLen={val_len:.2f}")

            # save last each N epochs
            if (epoch + 1) % max(1, args.save_every_epochs) == 0:
                saver.save_last(epoch, gstate["global_step"], best_val, history)

            # best & milestone
            if val_len < best_val:
                best_val = val_len; gstate["best_val"] = best_val
                best_payload = pack_ckpt(args, model, optimizer, epoch, gstate["global_step"], best_val, history)
                bestname = (
                    f"ddpbest_size_{out_dim}_hidden_{args.hidden_dim}_"
                    f"{args.optimizer}_tau_{args.tau}_n_iter_{args.n_iter}_noise_{args.noise_scale}_"
                    f"shift_{args.shift}_dist_scale_{args.distance_scale}_"
                    f"n_layers{args.n_layers}_seed_{args.seed}_dropout{args.dropout}.ckpt"
                )
                atomic_save(os.path.join(save_dir, bestname), best_payload)
                print(f"★ New best validation length: {val_len:.2f}")

            if (epoch + 1) % max(1, args.milestone_interval) == 0 and best_payload is not None:
                mname = (
                    f"ddpmilestone_best_up_to_epoch_{epoch+1}_"
                    f"size_{out_dim}_hidden_{args.hidden_dim}_"
                    f"{args.optimizer}_tau_{args.tau}_n_iter_{args.n_iter}_noise_{args.noise_scale}_"
                    f"shift_{args.shift}_dist_scale_{args.distance_scale}_"
                    f"n_layers{args.n_layers}_seed_{args.seed}_dropout{args.dropout}.ckpt"
                )
                atomic_save(os.path.join(save_dir, mname), best_payload)

            # basic instability warning
            if len(history["val_length"]) > 10:
                recent = history["val_length"][-10:]
                if max(recent) - min(recent) > 100.0:
                    print("[Warn] High val variance; consider smaller LR or enable grad clip.")

        barrier()

        # early stop
        if early and is_main() and early(val_len, (model.module if isinstance(model, DDP) else model)):
            print(f"[EarlyStop] at epoch {epoch+1}, best={best_val:.2f}")
            saver.save_last(epoch, gstate["global_step"], best_val, history)
            # broadcast stop via file flag for simplicity
            with open(os.path.join(save_dir, ".early_stop"), "w") as f:
                f.write("1")
        barrier()
        if os.path.isfile(os.path.join(save_dir, ".early_stop")):
            break

    if is_main():
        print("Training completed.")
        print(f"Best validation length: {best_val:.2f}")


if __name__ == "__main__":
    main()

