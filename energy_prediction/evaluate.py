import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# --------------------------- CONFIG ---------------------------
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

BINS = int((CONFIG["mz_max"] - CONFIG["mz_min"]) / CONFIG["bin_size"])


# --------------------- EVALUATION HELPERS ---------------------
def modified_cosine(pred, target, mz_tolerance=0.01, max_shift=200):
    """
    Compute modified cosine similarity.
    Accepts NumPy arrays OR PyTorch tensors.
    """
    # Convert to torch if NumPy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # Move to same device
    if pred.device != target.device:
        target = target.to(pred.device)

    batch_size, n_bins = pred.shape
    device = pred.device

    # Normalize spectra
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    target = target / (target.norm(dim=1, keepdim=True) + 1e-8)

    # Number of bins to shift
    shift_bins = int(max_shift / mz_tolerance)
    best_scores = torch.zeros(batch_size, device=device)

    for shift in range(-shift_bins, shift_bins + 1):
        if shift < 0:
            p = pred[:, :shift]
            t = target[:, -shift:]
        elif shift > 0:
            p = pred[:, shift:]
            t = target[:, :-shift]
        else:
            p = pred
            t = target

        if p.shape[1] == 0:
            continue

        # Dot product after shift
        dot = (p * t).sum(dim=1)
        best_scores = torch.max(best_scores, dot)

    return best_scores.cpu().numpy()  # Return as NumPy for consistency

def compute_cosine_similarities(pred, true):
    """Return list of cosine similarities (float)."""
    return [
        cosine_similarity(p.reshape(1, -1), t.reshape(1, -1))[0, 0]
        for p, t in zip(pred, true)
    ]

def compute_modified_cosine(pred, true, mz_tolerance=0.1, max_shift=200):
    pred_tensor = torch.from_numpy(pred)
    true_tensor = torch.from_numpy(true)
    return modified_cosine(pred_tensor, true_tensor, mz_tolerance, max_shift)

def group_by_ce(cosines, ces, indices):
    """Group cosine scores and original indices by starting CE."""
    groups = {0: [], 10: [], 40: []}
    idx_by_ce = {0: [], 10: [], 40: []}
    for i, (ce, sim) in enumerate(zip(ces, cosines)):
        c = int(ce)
        if c in groups:
            groups[c].append(sim)
            idx_by_ce[c].append(indices[i])
    return groups, idx_by_ce


def plot_ce_bar_chart(groups):
    """Bar chart with mean ± std."""
    avgs, stds, labels = [], [], []
    for ce, sims in sorted(groups.items()):
        if sims:
            avgs.append(np.mean(sims))
            stds.append(np.std(sims))
            labels.append(str(ce))

    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(labels, avgs, yerr=stds, capsize=8,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   error_kw={'elinewidth': 2, 'capthick': 2})
    plt.ylim(0, 1)
    plt.xlabel("Starting CE (eV)")
    plt.ylabel("Mean Cosine Similarity")
    plt.title("Performance per starting CE")
    for bar, avg in zip(bars, avgs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{avg:.3f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("images/ce_performance_breakdown.png", dpi=150)
    plt.close()
    print("Bar-chart → ce_performance_breakdown.png")


def plot_example_spectra(pred, true, ces, indices_by_ce, mz_axis, n_per_ce=2):
    """Plot n_per_ce random true vs. pred spectra per CE."""
    plt.figure(figsize=(15, 3 * len(indices_by_ce)))
    plot_idx = 0
    for ce in [0, 10, 40]:
        if ce not in indices_by_ce or len(indices_by_ce[ce]) == 0:
            continue
        sample_idxs = np.random.choice(
            indices_by_ce[ce], size=min(n_per_ce, len(indices_by_ce[ce])), replace=False
        )
        for i, idx in enumerate(sample_idxs):
            p, t = pred[idx], true[idx]
            sim = modified_cosine(
                p.reshape(1, -1), 
                t.reshape(1, -1), 
                mz_tolerance=CONFIG["bin_size"]
            )[0]  # ← returns NumPy array, so [0] works
            plt.subplot(len(indices_by_ce), n_per_ce, plot_idx + 1)
            plt.plot(mz_axis, t, label="True 20 eV", color="black", alpha=0.8)
            plt.plot(mz_axis, p, label="Pred", color="#d62728", alpha=0.8)
            plt.title(f"CE {ce}→20 | Cos: {sim:.3f}")
            plt.xlabel("m/z")
            plt.ylim(0, 1)
            plt.legend(fontsize=8)
            plot_idx += 1
    plt.tight_layout()
    plt.savefig("images/example_spectra_by_ce.png", dpi=150)
    plt.close()
    print("Example spectra → example_spectra_by_ce.png")

def evaluate_per_ce(model, X_s_val, X_c_val, y_val):
    model.eval()
    with torch.no_grad():
        pred = model(X_s_val, X_c_val).cpu().numpy()
        true = y_val.cpu().numpy()
        ces  = X_c_val.cpu().numpy().flatten() * CONFIG["max_ce"]
        indices = np.arange(len(ces))  # to retrieve original positions

    # 1. Cosine per sample
    cosines = compute_modified_cosine(pred, true, mz_tolerance=CONFIG["bin_size"])

    # 2. Group
    groups, idx_by_ce = group_by_ce(cosines, ces, indices)

    # 3. Print summary
    print("\n=== AVERAGE COSINE SIMILARITY (VALIDATION) ===")
    for ce, sims in sorted(groups.items()):
        if sims:
            print(f"  {ce} → 20 eV  |  n = {len(sims):5d}  |  "
                  f"mean = {np.mean(sims):.4f}  |  std = {np.std(sims):.4f}")
        else:
            print(f"  {ce} → 20 eV  |  n = 0")

    # 4. Plots
    mz_axis = np.linspace(CONFIG["mz_min"], CONFIG["mz_max"], BINS)
    plot_ce_bar_chart(groups)
    plot_example_spectra(pred, true, ces, idx_by_ce, mz_axis, n_per_ce=2)

def evaluate_overall(model, X_s_val, X_c_val, y_val):
    """
    Evaluate model performance across ALL validation samples.
    Prints mean/std cosine similarity and saves a histogram.
    """
    model.eval()
    with torch.no_grad():
        pred = model(X_s_val, X_c_val).cpu().numpy()
        true = y_val.cpu().numpy()

    # Compute cosine similarity for every pair
    cosines = compute_modified_cosine(pred, true, mz_tolerance=CONFIG["bin_size"])
    cosines = np.array(cosines)

    # Summary stats
    mean_cos = cosines.mean()
    std_cos  = cosines.std()
    n_total  = len(cosines)

    print("\n=== OVERALL PERFORMANCE (ALL CE → 20 eV) ===")
    print(f"  Total samples: {n_total}")
    print(f"  Mean Cosine:   {mean_cos:.4f}")
    print(f"  Std Dev:       {std_cos:.4f}")
    print(f"  Min / Max:     {cosines.min():.4f} / {cosines.max():.4f}")

    # Histogram of cosine similarities
    plt.figure(figsize=(7, 5))
    plt.hist(cosines, bins=50, range=(0, 1), color="#1f77b4", alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.axvline(mean_cos, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_cos:.3f}')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Spectra")
    plt.title("Distribution of Predicted vs True (20 eV) Cosine Similarity")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/overall_cosine_histogram.png", dpi=150)
    plt.close()
    print("Histogram → images/overall_cosine_histogram.png")