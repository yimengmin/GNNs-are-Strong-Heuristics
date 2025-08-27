import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

# Safe imports for compilation
try:
    import torch._dynamo
    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False

# Ultra-fast CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')  # Use Tensor cores
    print("🚀 CUDA Ultra-Fast Mode Enabled!")
    print("✓ cuDNN benchmark: True")
    print("✓ TF32 enabled: True") 
    print("✓ High precision matmul: True")
    print("✓ Tensor Core acceleration: Enabled")
else:
    print("⚠ CUDA not available, using CPU mode")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= Stability Utils =================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

def cycle_pos_encoding_fast(k: int, num_harmonics: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Ultra-fast cycle position encoding with fused operations"""
    two_pi = 2.0 * math.pi
    k_float = float(k)
    p = torch.arange(k, device=device, dtype=dtype)
    m = torch.arange(1, num_harmonics + 1, device=device, dtype=dtype)
    ang = two_pi * torch.outer(p, m) / k_float
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)

class CoordinateFeatureExtractor(nn.Module):
    """Fast coordinate feature extraction with polar features and position encoding"""
    def __init__(self, phi_harmonics=4, pos_harmonics=3, include_global_stats=True):
        super().__init__()
        self.phi_harmonics = phi_harmonics
        self.pos_harmonics = pos_harmonics
        self.include_global_stats = include_global_stats
        self.base_feat_dim = 3 + 2 * phi_harmonics
        self.pos_encoding_dim = 2 * pos_harmonics
        self.global_stats_dim = 8 if include_global_stats else 0
        self.total_feat_dim = self.base_feat_dim + self.pos_encoding_dim + self.global_stats_dim
        print(f"🎯 Coordinate Feature Extractor:")
        print(f"   • Polar harmonics: {phi_harmonics}")
        print(f"   • Position harmonics: {pos_harmonics}")
        print(f"   • Global stats: {include_global_stats}")
        print(f"   • Output feature dim: {self.total_feat_dim}")

    def polar_feats_fast(self, coords: torch.Tensor, c: torch.Tensor, 
                         u: torch.Tensor, u_perp: torch.Tensor) -> torch.Tensor:
        """Fast polar feature computation with fused operations"""
        B, N, _ = coords.shape
        x = coords - c.unsqueeze(1)               # [B,N,2]
        ax = torch.sum(x * u.unsqueeze(1), dim=-1)
        ay = torch.sum(x * u_perp.unsqueeze(1), dim=-1)
        r = torch.sqrt(ax.pow(2) + ay.pow(2) + 1e-6)
        theta = torch.atan2(ay, ax)
        m = torch.arange(1, self.phi_harmonics + 1, device=coords.device, dtype=coords.dtype)
        thm = theta.unsqueeze(-1) * m.unsqueeze(0).unsqueeze(0)
        feats = torch.cat([
            r.unsqueeze(-1), 
            ax.unsqueeze(-1), 
            ay.unsqueeze(-1),
            torch.sin(thm), 
            torch.cos(thm)
        ], dim=-1)
        return feats

    def compute_principal_axes(self, coords: torch.Tensor) -> tuple:
        """Stable principal axes via SVD per batch"""
        B, N, _ = coords.shape
        centers = coords.mean(dim=1, keepdim=True)        # [B,1,2]
        centered = coords - centers                       # [B,N,2]
        # SVD is numerically stabler than eigh for small 2x2 cov
        # centered = U S Vh, PCs are columns of V = Vh^T
        # Handle degenerate tiny batches gracefully
        Vh = torch.linalg.svd(centered, full_matrices=False).Vh   # [B,2,2]
        V = Vh.transpose(-1, -2)
        u = V[:, :, 0]       # [B,2]
        u_perp = V[:, :, 1]  # [B,2]
        return centers.squeeze(1), u, u_perp

    def compute_global_stats(self, coords: torch.Tensor) -> torch.Tensor:
        """Compute global coordinate statistics"""
        B, N, _ = coords.shape
        mean_coords = coords.mean(dim=1)                          # [B,2]
        std_coords = coords.std(dim=1, unbiased=False).clamp_min(1e-6)  # [B,2]
        min_coords = coords.min(dim=1)[0]
        max_coords = coords.max(dim=1)[0]
        bbox_size = (max_coords - min_coords).clamp_min(1e-6)
        bbox_center = (max_coords + min_coords) / 2
        global_stats = torch.cat([mean_coords, std_coords, bbox_size, bbox_center], dim=-1)  # [B,8]
        return global_stats

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B, N, _ = coords.shape
        device = coords.device
        dtype = coords.dtype
        centers, u, u_perp = self.compute_principal_axes(coords)
        polar_feats = self.polar_feats_fast(coords, centers, u, u_perp)  # [B,N,base]
        psi = cycle_pos_encoding_fast(N, self.pos_harmonics, device, dtype).unsqueeze(0).expand(B, -1, -1)
        features = torch.cat([polar_feats, psi], dim=-1)
        if self.include_global_stats:
            global_stats = self.compute_global_stats(coords).unsqueeze(1).expand(-1, N, -1)
            features = torch.cat([features, global_stats], dim=-1)
        return features

# ================= Safer Graph Diffusions =================
def fast_gcn_diffusion(W, order, feature):
    """GCN diffusion with degree clamp & no in-place ops"""
    batchsize, n, _ = W.shape
    A_gcn = W.clone()
    idx = torch.arange(n, device=W.device)
    A_gcn[:, idx, idx] = A_gcn[:, idx, idx] + 1.0
    degrees = A_gcn.sum(dim=2, keepdim=True).clamp_min(1e-6)
    D_inv_sqrt = torch.rsqrt(degrees)
    A_norm = A_gcn * D_inv_sqrt * D_inv_sqrt.transpose(-1, -2)
    results, x = [], feature
    for _ in range(order):
        x = torch.bmm(A_norm, x)
        results.append(F.gelu(x))
    return results

def fast_sct_diffusion(W, order, feature):
    """Row-stochastic power iteration with degree clamp"""
    degrees = W.sum(dim=2, keepdim=True).clamp_min(1e-6)
    D_inv = 1.0 / degrees
    P = W * D_inv.transpose(-1, -2)   # row-stochastic
    max_iter = 1 << order
    scale_points = [(1 << i) - 1 for i in range(order + 1)]
    x = feature
    snapshots = []
    for i in range(max_iter):
        x = 0.5 * (x + torch.bmm(P, x))
        if i in scale_points:
            snapshots.append(x)
    sct_features = [F.gelu(snapshots[i] - snapshots[i + 1]) for i in range(order)]
    return sct_features

# ================= Safer Sinkhorn =================
def fast_sinkhorn_iteration(Z, n_iter=10):
    """Float32 stabilized Sinkhorn"""
    Z = Z.to(torch.float32)
    for _ in range(n_iter):
        Z = Z - torch.logsumexp(Z, dim=-1, keepdim=True)  # rows
        Z = Z - torch.logsumexp(Z, dim=-2, keepdim=True)  # cols
    return Z

def fast_gumbel_sinkhorn(logits, tau=0.5, n_iter=10, noise_scale=0.05, clamp_val=20.0):
    """Stabilized Gumbel-Sinkhorn: milder tau, clamp logits, float32 compute"""
    logits = logits.clamp(-clamp_val, clamp_val)
    uniform = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-20) + 1e-20) * noise_scale
    Z_init = (logits.to(torch.float32) + gumbel_noise) / max(tau, 1e-3)
    Z_init = Z_init.clamp(-clamp_val, clamp_val)
    Z = fast_sinkhorn_iteration(Z_init, n_iter)
    P = F.softmax(Z, dim=-1)
    P = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-6)
    return P.to(logits.dtype), Z_init.to(logits.dtype)

# ================= Conv Block (PreNorm + Residual Scaling) =================
class UltraFastSCTConv(nn.Module):
    """Stable SCT conv: PreNorm, residual scale, no in-place, dropout"""
    def __init__(self, hidden_dim, sct_order=2, gcn_order=2, 
                 attn_dropout=0.05, ff_dropout=0.1, res_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sct_order = sct_order
        self.gcn_order = gcn_order
        self.num_channels = sct_order + gcn_order
        self.res_scale = res_scale

        self.norm_in = RMSNorm(hidden_dim)
        self.norm_post = RMSNorm(hidden_dim)

        self.attention_proj = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.proj1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ff_drop = nn.Dropout(ff_dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)

    def forward(self, X, adj, moment=1.0):
        Xin = self.norm_in(X)
        gcn_features = fast_gcn_diffusion(adj, self.gcn_order, Xin)
        sct_features = fast_sct_diffusion(adj, self.sct_order, Xin)
        if moment != 1.0:
            sct_features = [torch.abs(f).pow(moment) for f in sct_features]
        all_features = gcn_features + sct_features

        concat_features = torch.stack(
            [torch.cat([Xin, f], dim=-1) for f in all_features], dim=1
        )  # [B,C,N,2H]
        attn_logits = self.attention_proj(concat_features).squeeze(-1)  # [B,C,N]
        attn_logits = attn_logits / math.sqrt(self.hidden_dim)
        attn = F.softmax(attn_logits, dim=1)
        attn = self.attn_drop(attn)

        stacked = torch.stack(all_features, dim=1)       # [B,C,N,H]
        mixed = (attn.unsqueeze(-1) * stacked).sum(dim=1)

        out = F.gelu(self.proj1(self.norm_post(mixed)))
        out = self.proj2(out)
        out = self.ff_drop(out)

        return X + self.res_scale * out

# ================= GNN (Input/Final Norm, Learnable Scale, Clamp) =================
class EnhancedUltraFastSCTGNN(nn.Module):
    """Enhanced Ultra-optimized SCT-GNN with coordinate feature extraction"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, 
                 sct_order=6, gcn_order=1, tanh_scale=0.0, tau=0.5, 
                 n_iter=10, noise_scale=0.05, inference_mode=False,
                 coord_input=True, phi_harmonics=4, pos_harmonics=3, 
                 include_global_stats=True, attn_dropout=0.05, ff_dropout=0.1,
                 logit_clamp=20.0):
        super().__init__()
        self.coord_input = coord_input
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.tanh_scale = tanh_scale           # 默认 0，不再强行 tanh 压缩
        self.tau = tau
        self.n_iter = n_iter
        self.noise_scale = noise_scale
        self.inference = inference_mode
        self.logit_clamp = logit_clamp

        if coord_input:
            assert input_dim == 2, "Coordinate input must be 2D (x, y)"
            self.coord_extractor = CoordinateFeatureExtractor(
                phi_harmonics=phi_harmonics,
                pos_harmonics=pos_harmonics,
                include_global_stats=include_global_stats
            )
            actual_input_dim = self.coord_extractor.total_feat_dim
            print(f"🌟 Enhanced coordinate processing: {input_dim}D → {actual_input_dim}D")
        else:
            self.coord_extractor = None
            actual_input_dim = input_dim

        self.in_norm = RMSNorm(actual_input_dim)
        self.input_proj = nn.Linear(actual_input_dim, hidden_dim, bias=False)

        res_scale = 1.0 / math.sqrt(max(n_layers, 1))
        self.conv_layers = nn.ModuleList([
            UltraFastSCTConv(hidden_dim, sct_order, gcn_order, 
                             attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                             res_scale=res_scale)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(hidden_dim * (1 + n_layers))
        self.output_proj1 = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim, bias=False)
        self.output_proj2 = nn.Linear(hidden_dim, output_dim, bias=False)

        # learnable global scale for logits
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        print(f'🎯 Enhanced Ultra-Fast SCT-GNN initialized:')
        print(f'   • Coordinate input: {coord_input}')
        print(f'   • Input dimension: {input_dim} → {actual_input_dim}')
        print(f'   • SCT channels: {sct_order}')
        print(f'   • GCN channels: {gcn_order}') 
        print(f'   • Total layers: {n_layers}')
        print(f'   • Residual scale: {res_scale:.3f}')
        print(f'   • Parameters: {sum(p.numel() for p in self.parameters()):,}')

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)

    def _get_logits(self, X, adj, moment=1.0):
        if self.coord_input and self.coord_extractor is not None:
            X = self.coord_extractor(X)
        X = self.input_proj(self.in_norm(X))

        states = [X]
        for conv in self.conv_layers:
            X = conv(X, adj, moment)
            states.append(X)

        Xall = torch.cat(states, dim=-1)
        Xall = self.final_norm(Xall)

        h = F.gelu(self.output_proj1(Xall))
        logits = self.output_proj2(h) * self.logit_scale

        # optional tanh squeezing (default 0, i.e., disabled)
        if self.tanh_scale and self.tanh_scale > 0:
            logits = torch.tanh(logits) * self.tanh_scale

        # clamp to avoid softmax/sinkhorn overflow
        logits = logits.clamp(-self.logit_clamp, self.logit_clamp)
        return logits

    def forward(self, X, adj, moment=1.0):
        logits = self._get_logits(X, adj, moment)
        if self.training and not self.inference:
            P, Z_init = fast_gumbel_sinkhorn(
                logits, tau=self.tau, n_iter=self.n_iter, noise_scale=self.noise_scale,
                clamp_val=self.logit_clamp
            )
            return P, Z_init
        else:
            return logits
            #print('inference:-------')
            #noise = -torch.log(-torch.log(torch.rand(logits.shape, device=logits.device) + 1e-20) + 1e-20)
            #noise = noise*self.noise_scale   # Scale noise too
            #return logits + noise

def create_enhanced_ultra_fast_sct_gnn(input_dim, hidden_dim, output_dim, n_layers=3,
                                       sct_order=6, gcn_order=1, compile_model=False, 
                                       coord_input=False, phi_harmonics=4, pos_harmonics=3,
                                       include_global_stats=True, **kwargs):
    """Create enhanced ultra-optimized SCT-GNN with coordinate processing"""
    print("🚀 Creating Enhanced Ultra-Fast SCT-GNN...")
    model = EnhancedUltraFastSCTGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim, 
        output_dim=output_dim,
        n_layers=n_layers,
        sct_order=sct_order,
        gcn_order=gcn_order,
        coord_input=coord_input,
        phi_harmonics=phi_harmonics,
        pos_harmonics=pos_harmonics,
        include_global_stats=include_global_stats,
        **kwargs
    )
    if compile_model and torch.cuda.is_available():
        try:
            print("⚡ Attempting model compilation...")
            test_input = torch.randn(2, 10, input_dim, device='cuda', requires_grad=False)
            test_adj = torch.rand(2, 10, 10, device='cuda', requires_grad=False)
            model = model.to('cuda')
            model.eval()
            with torch.no_grad():
                _ = model(test_input, test_adj)
            model = torch.compile(model, mode="default")
            print("✓ Model compilation successful!")
        except Exception as e:
            print(f"⚠ Compilation failed: {e}")
            print("⚠ Running without compilation (still optimized!)")
    elif compile_model:
        print("⚠ CUDA not available, skipping compilation")
    else:
        print("⚠ Compilation disabled - use manual compilation in training script if needed")
    return model

# Aliases for compatibility
GNN = EnhancedUltraFastSCTGNN
create_ultra_fast_sct_gnn = create_enhanced_ultra_fast_sct_gnn

@torch.no_grad()
def test_coordinate_enhancement():
    """Test coordinate feature enhancement"""
    print("🧪 Testing Enhanced Coordinate Processing")
    print("=" * 50)
    B, N = 4, 20
    coords = torch.randn(B, N, 2, device=device) * 10
    model = create_enhanced_ultra_fast_sct_gnn(
        input_dim=2, hidden_dim=64, output_dim=N,
        n_layers=2, sct_order=3, gcn_order=1,
        coord_input=True, phi_harmonics=3, pos_harmonics=2,
        include_global_stats=True, tau=0.5, n_iter=10
    ).to(device)
    adj = torch.rand(B, N, N, device=device)
    adj = (adj + adj.transpose(-1, -2)) / 2
    print(f"Input coordinates shape: {coords.shape}")
    print("\n🔍 Testing coordinate feature extraction:")
    enhanced_feats = model.coord_extractor(coords)
    print(f"Enhanced features shape: {enhanced_feats.shape}")
    ce = model.coord_extractor
    print(f"   • Base polar features: {ce.base_feat_dim}")
    print(f"   • Position encoding: {ce.pos_encoding_dim}")
    print(f"   • Global statistics: {ce.global_stats_dim}")
    print(f"   • Total features: {ce.total_feat_dim}")
    print(f"\n🚀 Testing full model forward pass:")
    output = model(coords, adj)
    if isinstance(output, tuple):
        P, Z = output
        print(f"Output shapes: P={P.shape}, Z={Z.shape}")
        print(f"Doubly stochastic check: rows={P.sum(-1).mean():.4f}, cols={P.sum(-2).mean():.4f}")
    else:
        print(f"Output shape: {output.shape}")
    print("✓ Enhanced coordinate processing test successful!")

if __name__ == "__main__":
    print("🚀 Enhanced Ultra-Fast SCT-GNN with Coordinate Processing")
    print("=" * 60)
    if DYNAMO_AVAILABLE:
        torch._dynamo.config.suppress_errors = True
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        print(f"✓ Using device: {device}")
    else:
        print("⚠ CUDA not available, using CPU")
    try:
        test_coordinate_enhancement()
    except Exception as e:
        print(f"❌ Enhanced coordinate test failed: {e}")
        import traceback; traceback.print_exc()
    print("\n🎉 Enhanced SCT-GNN testing complete!")
    print("\n💡 Enhanced features:")
    print("   • PreNorm (RMSNorm) + residual scaling")
    print("   • Degree clamp in GCN/SCT")
    print("   • Attention √d scaling + Dropout")
    print("   • Float32 Sinkhorn + logits clamp")
    print("   • SVD-based principal axes")
    print("   • Learnable logit scale (no forced tanh)")

