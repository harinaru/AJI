import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Optional

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, PolynomialFeatures, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

logging.basicConfig(level=logging.INFO)

DATA_PATH = r"D:\Codings\AJi Final\results\phase2_features_fixed"
TEST_SIZE = 0.2


# =========================
# FILE SELECT
# =========================
def select_best_file(files: List[str]) -> Optional[str]:
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
    common: Optional[set] = None

    for p in tqdm(os.listdir(DATA_PATH), desc="Scanning patients"):
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
def load_data(cols: List[str]):
    X, y, pid = [], [], []

    for p in tqdm(os.listdir(DATA_PATH), desc="Loading data"):
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
        y[np.isin(pid, te)]
    )


# =========================
# MAIN
# =========================
def main():

    logging.info("🚀 Starting pipeline...")

    cols = get_common_cols()
    X, y, pid = load_data(cols)

    logging.info(f"Class distribution: {np.bincount(y)}")

    Xtr, Xte, ytr, yte = patient_split(X, y, pid)

    # Scaling
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # Feature selection
    selector = SelectKBest(f_classif, k=40)
    Xtr = selector.fit_transform(Xtr, ytr)
    Xte = selector.transform(Xte)

    # Feature interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    Xtr = poly.fit_transform(Xtr)
    Xte = poly.transform(Xte)

    # Normalize
    scaler2 = StandardScaler()
    Xtr = scaler2.fit_transform(Xtr)
    Xte = scaler2.transform(Xte)

    # Models
    models = {
        "LR": LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', n_jobs=-1),
        "RF": RandomForestClassifier(n_estimators=800, max_depth=20, class_weight='balanced', n_jobs=-1),
        "GB": HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03),
        "MLP": MLPClassifier((512,256,128), max_iter=800)
    }

    results = {}
    best_model = None
    best_f1 = 0.0

    # Train
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

    # Ensemble
    ens = VotingClassifier([(k, v) for k, v in models.items()], voting='soft')
    ens.fit(Xtr, ytr)
    preds = ens.predict(Xte)
    results["ENS"] = f1_score(yte, preds, average='macro')

    # =========================
    # FIXED BAR PLOT
    # =========================
    plt.figure()
    plt.bar(list(results.keys()), list(results.values()))
    plt.xticks(rotation=30)
    plt.title("Model Comparison (F1)")
    plt.show()

    # =========================
    # FIXED ROC
    # =========================
    if best_model is not None and hasattr(best_model, "predict_proba"):
        y_bin = label_binarize(yte, classes=np.unique(y))
        y_score = best_model.predict_proba(Xte)

        if y_score is not None:
            plt.figure()
            for i in range(y_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                plt.plot(fpr, tpr, label=f"Class {i} AUC={auc(fpr, tpr):.2f}")

            plt.legend()
            plt.title("ROC Curve")
            plt.show()

    logging.info(f"Final Results: {results}")


if __name__ == "__main__":
    main()