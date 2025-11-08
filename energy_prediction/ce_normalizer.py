# ------------------------------------------------------------
#  FILE: ce_normalizer_modular.py
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from evaluate import evaluate_per_ce, evaluate_overall
import yaml

# -------------------------- CONFIG --------------------------
print("Loading config...")
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
# -----------------------------------------------------------

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

BINS = int((CONFIG["mz_max"] - CONFIG["mz_min"]) / CONFIG["bin_size"])

# ----------------------- HELPERS ---------------------------
def parse_and_bin(spec):
    mz = spec[:, 0].astype(float)
    intens = spec[:, 1].astype(float)
    indices = ((mz - CONFIG["mz_min"]) / CONFIG["bin_size"]).astype(int)
    valid = (indices >= 0) & (indices < BINS)
    vec = np.zeros(BINS)
    np.add.at(vec, indices[valid], intens[valid])
    return vec / vec.max() if vec.max() > 0 else vec
import torch
import torch.nn.functional as F


# ----------------------- DATA LOADER -----------------------
def load_and_prepare_data():
    print("Loading data...")
    df = pd.read_pickle(CONFIG["file_path"])

    # 20 eV lookup (InChI → binned vector)
    print("Building 20 eV lookup...")
    target_specs = {}
    for _, row in df[df['collision_energy'] == CONFIG["target_ce"]].iterrows():
        target_specs[row['inchi']] = parse_and_bin(row['spectrum'])

    # training pairs (spec@CE, CE, spec@20)
    print("Building training pairs...")
    data = []
    for _, row in df[df['collision_energy'] != CONFIG["target_ce"]].iterrows():
        if row['inchi'] not in target_specs:
            continue
        data.append({
            'spec_in': parse_and_bin(row['spectrum']),
            'ce_norm': row['collision_energy'] / CONFIG["max_ce"],
            'target': target_specs[row['inchi']],
        })

    print(f"Total training pairs: {len(data)}")
    return data


# -------------------------- MODEL --------------------------
class Spec2Spec(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BINS + 1, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, BINS),
            nn.Sigmoid()
        )


    def forward(self, s, c):
        x = torch.cat([s, c], dim=1)
        return self.net(x)


# -------------------------- TRAIN --------------------------
def train_model(train_loader, val_x_s, val_x_c, val_y):
    model = Spec2Spec()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    print("Training...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        for spec_b, ce_b, target_b in train_loader:
            optimizer.zero_grad()
            pred = model(spec_b, ce_b)
            cos_sim = nn.functional.cosine_similarity(pred, target_b, dim=1)
            loss = 1 - cos_sim.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # validation cosine
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_s, val_x_c)
            cos_sim = nn.functional.cosine_similarity(val_pred, val_y, dim=1).mean().item()
        loss = 1 - cos_sim

        if epoch % 10 == 0 or epoch == CONFIG["epochs"] - 1:
            print(f"Epoch {epoch:2d} | TrainLoss {epoch_loss/len(train_loader):.5f} | ValCos {cos_sim:.5f}")

    torch.save(model.state_dict(), CONFIG["model_path"])
    print(f"Model saved → {CONFIG['model_path']}")
    return model


# -------------------------- MAIN --------------------------
def main(retrain: bool = False):
    # data
    data = load_and_prepare_data()

    # tensors
    X_spec    = torch.from_numpy(np.array([d['spec_in'] for d in data])).float()
    X_ce_norm = torch.tensor([[d['ce_norm']] for d in data], dtype=torch.float32)
    y_spec    = torch.from_numpy(np.array([d['target'] for d in data])).float()

    # split
    X_s_tr, X_s_val, X_c_tr, X_c_val, y_tr, y_val = train_test_split(
        X_spec, X_ce_norm, y_spec, test_size=0.2, random_state=CONFIG["seed"]
    )

    # training
    model_path = Path(CONFIG["model_path"])
    if retrain or not model_path.exists():
        train_loader = DataLoader(
            TensorDataset(X_s_tr, X_c_tr, y_tr),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
        )
        model = train_model(train_loader, X_s_val, X_c_val, y_val)
    else:
        print(f"Loading existing model → {model_path}")
        model = Spec2Spec()
        model.load_state_dict(torch.load(model_path))
        model.eval()

    # evaluation / plotting
    evaluate_per_ce(model, X_s_val, X_c_val, y_val)
    evaluate_overall(model, X_s_val, X_c_val, y_val)


if __name__ == "__main__":
    # Run once with training:
    main(retrain=False)