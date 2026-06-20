"""
verify.py
=========

Empirical verification of Theorem 4: Attention Entropy Barrier.

We verify three parts:
    1. The closed-form barrier formula is exact for 1-vs-all logits.
    2. The inverse formula recovers minimum gap for target entropy.
    3. A training transformer follows the (Δ, H) barrier trajectory.

Requirements: NumPy, PyTorch (GPU on Jetson)

Usage:
    source ~/heartlib/.venv/bin/activate
    python empirical/verify.py
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Use PyTorch for GPU + training
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Section 1: Closed-form barrier formula
# =============================================================================

def barrier_formula(delta: float, T: float, d: int) -> float:
    """Exact entropy for 1-vs-all logits with dominant gap Δ.

    H(Δ, T, d) = log(exp(Δ/T) + d - 1)
                   - (Δ/T * exp(Δ/T)) / (exp(Δ/T) + d - 1)
    """
    et = math.exp(delta / T)
    S = et + (d - 1)
    return math.log(S) - (delta / T) * et / S


def entropy_softmax(z: np.ndarray, T: float = 1.0) -> float:
    """Compute softmax entropy from logits z."""
    z_norm = z / T
    z_max = z_norm.max()
    exp_z = np.exp(z_norm - z_max)
    p = exp_z / exp_z.sum()
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    return float(-np.sum(p_safe * np.log(p_safe)))


def compute_delta_gap(z: np.ndarray) -> float:
    """Dominant logit gap: z_max - mean(z)."""
    return float(z.max() - z.mean())


def make_one_vs_all_logits(delta: float, d: int) -> np.ndarray:
    """Logits: [delta, 0, 0, ..., 0] (d elements)."""
    z = np.zeros(d)
    z[0] = delta
    return z


# =============================================================================
# Section 2: Inverse barrier (find Δ for target H)
# =============================================================================

def invert_barrier(H_target: float, T: float, d: int,
                   delta_min: float = 0.0, delta_max: float = 50.0,
                   tol: float = 1e-6) -> float:
    """Bisection search for Δ such that H_barrier(Δ) = H_target."""
    lo, hi = delta_min, delta_max
    f_lo = barrier_formula(lo, T, d)
    f_hi = barrier_formula(hi, T, d)

    # Check target is in range
    if H_target >= f_lo:
        return 0.0
    if H_target <= f_hi:
        return hi

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        f_mid = barrier_formula(mid, T, d)
        if f_mid > H_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# =============================================================================
# Section 3: Minimal transformer for training trajectory
# =============================================================================

class MinimalTransformer(nn.Module):
    """Single-layer, single-head transformer for trajectory tracking."""

    def __init__(self, vocab_size: int, d_model: int, seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor):
        """Returns logits, attention_probs, attn_logits."""
        # tokens: (batch, seq_len)
        x = self.embed(tokens)  # (batch, seq_len, d_model)
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq, seq)
        attn_logits = scores / math.sqrt(self.d_model)
        attn_probs = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_probs, V)
        logits = self.proj(out)

        return logits, attn_probs, attn_logits


def generate_copy_first_data(n_samples: int, vocab_size: int, seq_len: int, device: torch.device):
    """Generate data: predict first token from sequence.
    Input: [t, random, random, ...]
    Target: first token repeated at all positions.
    """
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
    # Target: first token at all positions (copy task)
    targets = tokens[:, 0:1].expand(n_samples, seq_len)
    return tokens, targets


def extract_delta_entropy(attn_logits: torch.Tensor) -> Tuple[float, float]:
    """Extract Δ (dominant gap) and H (entropy) from a single attention head.

    attn_logits: (seq_len, seq_len) for a single sample/head
    """
    n = attn_logits.shape[0]
    # Use the attention distribution for the last position (most constrained)
    z = attn_logits[-1, :].detach().cpu().numpy()
    delta = compute_delta_gap(z)
    H = entropy_softmax(z, T=1.0)
    return delta, H


# =============================================================================
# Section 4: Theorem checks
# =============================================================================

@dataclass
class TheoremResult:
    name: str
    passed: bool
    metric: float
    detail: str


def check_theorem_4_1() -> TheoremResult:
    """Theorem 4.1: Barrier formula exactness."""
    test_cases = []
    max_err = 0.0

    for d in [4, 8, 16, 32]:
        for T in [0.5, 1.0, 2.0]:
            for delta in np.linspace(0, 20, 21):
                z = make_one_vs_all_logits(delta, d)
                H_formula = barrier_formula(delta, T, d)
                H_empirical = entropy_softmax(z, T)
                err = abs(H_formula - H_empirical)
                max_err = max(max_err, err)
                if len(test_cases) < 5 and err > 1e-10:
                    test_cases.append(f"d={d}, T={T}, Δ={delta:.1f}: H_form={H_formula:.6f}, H_emp={H_empirical:.6f}, err={err:.2e}")

    passed = max_err < 1e-8
    return TheoremResult(
        name="Theorem 4.1: Barrier Formula Exactness",
        passed=passed,
        metric=max_err,
        detail=f"Max error={max_err:.2e} across {4*3*21} cases. Samples: {'; '.join(test_cases[:3])}",
    )


def check_theorem_4_2() -> TheoremResult:
    """Theorem 4.2: Inverse barrier (minimum gap for target entropy)."""
    d = 8
    T = 1.0
    max_err = 0.0
    samples = []

    for H_star in np.linspace(0.1, math.log(d) - 0.1, 10):
        delta_inv = invert_barrier(H_star, T, d)
        H_check = barrier_formula(delta_inv, T, d)
        err = abs(H_check - H_star)
        max_err = max(max_err, err)
        if len(samples) < 3:
            samples.append(f"H*={H_star:.3f}: Δ_inv={delta_inv:.3f}, H_check={H_check:.3f}, err={err:.2e}")

        # Verify monotonicity: smaller gap gives higher entropy
        delta_small = max(0, delta_inv - 1.0)
        H_small = barrier_formula(delta_small, T, d)
        if H_small <= H_star:
            passed_mono = False
        else:
            passed_mono = True

    passed = (max_err < 1e-4) and passed_mono
    return TheoremResult(
        name="Theorem 4.2: Inverse Barrier (Min Gap)",
        passed=passed,
        metric=max_err,
        detail=f"Max inversion error={max_err:.2e}; monotonicity={'PASS' if passed_mono else 'FAIL'}; samples: {'; '.join(samples)}",
    )


def check_theorem_4_3(device: torch.device) -> TheoremResult:
    """Theorem 4.3: Training trajectory crosses barrier.

    Train a minimal transformer on "copy first token" task.
    Track (Δ, H) at each step and verify trajectory follows barrier.
    """
    if not HAS_TORCH:
        return TheoremResult(
            name="Theorem 4.3: Training Trajectory",
            passed=False,
            metric=0.0,
            detail="PyTorch not available",
        )

    vocab_size = 4
    d_model = 8
    seq_len = 4
    n_train = 256
    batch_size = 32
    n_steps = 200

    model = MinimalTransformer(vocab_size, d_model, seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)

    trajectory = []  # list of (step, delta, H)

    for step in range(n_steps):
        tokens, targets = generate_copy_first_data(batch_size, vocab_size, seq_len, device)
        opt.zero_grad()
        logits, attn_probs, attn_logits = model(tokens)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        opt.step()

        # Extract Δ and H from last position's attention
        with torch.no_grad():
            logits_val, attn_probs_val, attn_logits_val = model(tokens[:1])
            delta, H = extract_delta_entropy(attn_logits_val[0])

        trajectory.append((step, delta, H, loss.item()))

    # Check trajectory: entropy should decrease while Δ increases
    deltas = [t[1] for t in trajectory]
    entropies = [t[2] for t in trajectory]

    # Compute correlation between Δ and H (should be negative)
    if len(deltas) > 10:
        corr = np.corrcoef(deltas[10:], entropies[10:])[0, 1]
    else:
        corr = 0.0

    # Check barrier fit: compute mean squared error between empirical H and barrier H
    mse = 0.0
    d = seq_len
    T = 1.0
    for _, delta, H_emp, _ in trajectory[50:]:
        if delta < 50:
            H_pred = barrier_formula(delta, T, d)
            mse += (H_pred - H_emp) ** 2
    mse = mse / max(1, len(trajectory) - 50)

    # Core evidence: trajectory follows barrier curve with negative correlation
    delta_initial = deltas[0]
    H_initial = entropies[0]
    delta_final = deltas[-1]
    H_final = entropies[-1]
    passed = (corr < -0.5) and (mse < 0.01)

    return TheoremResult(
        name="Theorem 4.3: Training Trajectory (Barrier Tracking)",
        passed=passed,
        metric=corr,
        detail=(f"Initial: Δ={delta_initial:.3f}, H={H_initial:.3f} | "
                f"Final: Δ={delta_final:.3f}, H={H_final:.3f} | "
                f"Corr(Δ,H)={corr:.3f} (target < -0.5) | "
                f"Barrier-MSE={mse:.6f} (target < 0.01) | Loss={trajectory[-1][3]:.4f}"),
    )


# =============================================================================
# Section 5: Main runner
# =============================================================================

def main() -> int:
    print("=" * 70)
    print(" Theorem 4: Attention Entropy Barrier")
    print(" Empirical Verification")
    print("=" * 70)

    np.random.seed(1729)
    if HAS_TORCH:
        torch.manual_seed(1729)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = None
    print()

    results = []

    print("--- Theorem 4.1: Barrier Formula Exactness ---")
    r1 = check_theorem_4_1()
    results.append(r1)
    print(f"\n{'PASS' if r1.passed else 'FAIL'} — {r1.name}")
    print(f"  Metric: {r1.metric}")
    print(f"  Detail: {r1.detail}")

    print("\n--- Theorem 4.2: Inverse Barrier ---")
    r2 = check_theorem_4_2()
    results.append(r2)
    print(f"\n{'PASS' if r2.passed else 'FAIL'} — {r2.name}")
    print(f"  Metric: {r2.metric}")
    print(f"  Detail: {r2.detail}")

    print("\n--- Theorem 4.3: Training Trajectory ---")
    r3 = check_theorem_4_3(device)
    results.append(r3)
    print(f"\n{'PASS' if r3.passed else 'FAIL'} — {r3.name}")
    print(f"  Metric: {r3.metric}")
    print(f"  Detail: {r3.detail}")

    n_pass = sum(1 for r in results if r.passed)
    print("\n" + "=" * 70)
    print(f"OVERALL: {n_pass}/{len(results)} theorems verified")
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"  {flag} — {r.name}")
    print("=" * 70)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
