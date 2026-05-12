import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score, roc_curve

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)

DATA_PATH = r"D:\Codings\AJi Final\results\phase2_features_fixed"
TEST_SIZE = 0.2
SEQ_LEN = 20


# =========================
# FILE SELECT
# =========================
def select_best_file(files: List[str]) -> str | None:
    for k in ["B150", "B10", "B183"]:
        for f in files:
            if k in f:
                return f
    return None


# =========================
# CLEAN CAP
# =========================
def clean_cap(df: pd.DataFrame) -> pd.DataFrame:
    cap_map = {'B': 0, 'A1': 1, 'A2': 2, 'A3': 3}
    df['cap_subtype'] = df.get('cap_subtype', 'B')
    df['cap_subtype'] = df['cap_subtype'].fillna('B')
    df['cap_subtype'] = df['cap_subtype'].apply(lambda x: cap_map.get(x, 0)).astype(int)
    return df


# =========================
# COMMON FEATURES
# =========================
def get_common_cols() -> List[str]:
    common = None

    for p in tqdm(os.listdir(DATA_PATH)):
        path = os.path.join(DATA_PATH, p)
        if not os.path.isdir(path):
            continue

        f = select_best_file(os.listdir(path))
        if f is None:
            continue

        df = clean_cap(pd.read_csv(os.path.join(path, f)))
        cols = set(df.columns) - {'cap_subtype'}

        common = cols if common is None else common & cols

    if common is None:
        raise ValueError("No common features found")

    return sorted(list(common))


# =========================
# LOAD DATA
# =========================
def load_data(cols):
    X, y, pid = [], [], []

    for p in tqdm(os.listdir(DATA_PATH)):
        path = os.path.join(DATA_PATH, p)
        if not os.path.isdir(path):
            continue

        f = select_best_file(os.listdir(path))
        if f is None:
            continue

        df = clean_cap(pd.read_csv(os.path.join(path, f)))
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty:
            continue

        X.append(df[cols].values)
        y.append(df['cap_subtype'].values)
        pid.extend([p] * len(df))

    if len(X) == 0:
        raise ValueError("No valid data loaded")

    return np.vstack(X), np.concatenate(y), np.array(pid)


# =========================
# SPLIT
# =========================
def patient_split(X, y, pid):
    pts = np.unique(pid)
    tr, te = train_test_split(pts, test_size=TEST_SIZE, random_state=42)

    return (
        X[np.isin(pid, tr)],
        X[np.isin(pid, te)],
        y[np.isin(pid, tr)],
        y[np.isin(pid, te)],
        pid[np.isin(pid, tr)],
        pid[np.isin(pid, te)],
    )


# =========================
# TEMPORAL DATASET
# =========================
class EEGDataset(Dataset):
    def __init__(self, X, y, pid):
        self.samples = []

        for p in np.unique(pid):
            idx = np.where(pid == p)[0]

            if len(idx) < SEQ_LEN:
                continue

            for i in range(len(idx) - SEQ_LEN):
                self.samples.append((
                    X[idx[i:i+SEQ_LEN]],
                    y[idx[i+SEQ_LEN]]
                ))

        if len(self.samples) == 0:
            raise ValueError("No temporal sequences created")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)


# =========================
# CAP-LTC MODEL
# =========================
class CAP_LTC(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 64)
        self.norm = nn.LayerNorm(64)
        self.out = nn.Linear(64, 4)

        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.tau0 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        h = torch.zeros(x.size(0), 64).to(x.device)

        for t in range(x.size(1)):
            xt = x[:, t, :]
            cap = torch.sigmoid(xt[:, -1])

            tau = self.tau0 * torch.exp(self.alpha * cap)
            tau = torch.clamp(tau, 0.1, 10)

            dh = -h / tau.unsqueeze(1) + torch.tanh(self.fc(xt))
            h = self.norm(h + 0.1 * dh)

        return self.out(h)


# =========================
# TRAIN LTC
# =========================
def train_ltc(Xtr, ytr, Xte, yte, ptr, pte):

    try:
        tr_ds = EEGDataset(Xtr, ytr, ptr)
        te_ds = EEGDataset(Xte, yte, pte)
    except Exception as e:
        logging.warning(f"LTC skipped: {e}")
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=64)

    model = CAP_LTC(Xtr.shape[1]).to(device)

    weights = compute_class_weight('balanced', classes=np.unique(ytr), y=ytr)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(12):
        for xb, yb in tqdm(tr_dl, desc=f"LTC {ep}"):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    preds, true = [], []

    model.eval()
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            p = torch.argmax(model(xb), dim=1).cpu().numpy()
            preds.extend(p)
            true.extend(yb.numpy())

    return f1_score(true, preds, average='macro')


# =========================
# MAIN
# =========================
def main():

    logging.info("🚀 Starting pipeline...")

    cols = get_common_cols()
    X, y, pid = load_data(cols)

    mask = y > 0
    X, y, pid = X[mask], y[mask], pid[mask]

    Xtr, Xte, ytr, yte, ptr, pte = patient_split(X, y, pid)

    # Scaling
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # Feature selection
    selector = VarianceThreshold(threshold=0.01)
    Xtr = selector.fit_transform(Xtr)
    Xte = selector.transform(Xte)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    Xtr_cap = np.concatenate((Xtr, ytr.reshape(-1, 1)), axis=1)
    Xte_cap = np.concatenate((Xte, yte.reshape(-1, 1)), axis=1)

    models = {
        "LR": LogisticRegression(max_iter=5000, solver='saga', n_jobs=-1, class_weight='balanced'),
        "RF": RandomForestClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1),
        "GB": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05),
        "MLP": MLPClassifier((512,256,128), max_iter=800)
    }

    results = {}
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        logging.info(f"Training {name}")
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)

        f1 = f1_score(yte, preds, average='macro')
        results[name] = f1
        logging.info(f"{name}: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    ens = VotingClassifier([(k, v) for k, v in models.items()], voting='soft')
    ens.fit(Xtr, ytr)
    preds = ens.predict(Xte)
    results["ENS"] = f1_score(yte, preds, average='macro')

    results["CAP-LTC"] = train_ltc(Xtr_cap, ytr, Xte_cap, yte, ptr, pte)

    logging.info(f"Final Results: {results}")

    plt.bar(list(results.keys()), list(results.values()))
    plt.xticks(rotation=30)
    plt.title("Final Model Comparison")
    plt.show()


if __name__ == "__main__":
    main()