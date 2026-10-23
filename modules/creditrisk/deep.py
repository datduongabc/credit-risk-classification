"""Deep learning pipeline: an MLP with entity embeddings (PyTorch) and TabNet.

Both networks learn their own representation end to end, so preprocessing is limited to
what the optimizer needs: imputation and standardization for numeric columns and integer
codes for categorical columns, which are fed to trainable embedding tables.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn

from . import config
from .evaluation import compute_metrics, f2_threshold
from .transformers import ENGINEERED_FEATURES, GroupMedianImputer, IQRClipper, feature_engineer


# Deep results move slightly between hardware and PyTorch builds (early stopping can halt at a
# different epoch), so each network is trained once per seed and reported as mean +- sd.
SEEDS = tuple(config.SEED + i for i in range(5))


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TabularEncoder:
    """Numeric block (imputed, clipped, standardized) plus integer codes for categoricals.

    Code 0 is reserved for categories unseen during fitting, so embeddings never index out
    of range on new data.
    """

    def __init__(self, feature_engineering: bool = True):
        self.feature_engineering = feature_engineering

    def fit(self, X: pd.DataFrame):
        numeric = list(config.NUMERIC_FEATURES) + (ENGINEERED_FEATURES if self.feature_engineering else [])
        steps = [("group_impute", GroupMedianImputer())]
        if self.feature_engineering:
            steps.append(("engineer", feature_engineer()))
        num_pipe = SkPipeline(
            [("clip", IQRClipper()), ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        cat_pipe = SkPipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        steps.append(
            ("columns", ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, config.CATEGORICAL_FEATURES)]))
        )
        self.pipe_ = SkPipeline(steps).fit(X)
        self.n_numeric_ = len(numeric)
        self.numeric_names_ = numeric
        cats = self.pipe_.named_steps["columns"].named_transformers_["cat"].named_steps["ordinal"].categories_
        self.cardinalities_ = [len(c) + 1 for c in cats]
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Z = self.pipe_.transform(X)
        num = Z[:, : self.n_numeric_].astype(np.float32)
        cat = (Z[:, self.n_numeric_ :] + 1).astype(np.int64)
        return num, cat


class EmbeddingMLP(nn.Module):
    """Categoricals -> embeddings, concatenated with numerics -> MLP -> default logit."""

    def __init__(self, n_numeric, cardinalities, hidden=(128, 64), dropout=0.2, use_embeddings=True):
        super().__init__()
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.embeddings = nn.ModuleList(nn.Embedding(c, min(8, (c + 1) // 2)) for c in cardinalities)
            cat_dim = sum(e.embedding_dim for e in self.embeddings)
        else:
            # One-hot baseline: a frozen identity embedding is exactly one-hot encoding.
            self.embeddings = nn.ModuleList(nn.Embedding.from_pretrained(torch.eye(c), freeze=True) for c in cardinalities)
            cat_dim = sum(cardinalities)
        layers, d = [], n_numeric + cat_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self.repr_dim = d

    def represent(self, num, cat):
        emb = [e(cat[:, i]) for i, e in enumerate(self.embeddings)]
        return self.body(torch.cat([num, *emb], dim=1))

    def forward(self, num, cat):
        return self.head(self.represent(num, cat)).squeeze(1)


@dataclass
class MLPConfig:
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    use_embeddings: bool = True
    pos_weight: bool = True
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    feature_engineering: bool = True
    name: str = field(default="MLP")

    def as_dict(self):
        return asdict(self)


class MLPClassifier:
    """Trains an EmbeddingMLP with early stopping on validation PR-AUC."""

    def __init__(self, cfg: MLPConfig | None = None, seed: int = config.SEED):
        self.cfg = cfg or MLPConfig()
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):
        set_seed(self.seed)
        cfg = self.cfg
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.seed)
        self.encoder_ = TabularEncoder(cfg.feature_engineering).fit(X_tr)
        num_tr, cat_tr = map(torch.from_numpy, self.encoder_.transform(X_tr))
        num_val, cat_val = map(torch.from_numpy, self.encoder_.transform(X_val))
        yt = torch.tensor(np.asarray(y_tr), dtype=torch.float32)
        yv = np.asarray(y_val)

        self.model_ = EmbeddingMLP(
            self.encoder_.n_numeric_, self.encoder_.cardinalities_, cfg.hidden, cfg.dropout, cfg.use_embeddings
        )
        pw = torch.tensor((len(yt) - yt.sum()) / yt.sum()) if cfg.pos_weight else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

        best, best_state, wait, self.history_ = -np.inf, None, 0, []
        gen = torch.Generator().manual_seed(self.seed)
        t0 = time.perf_counter()
        for epoch in range(cfg.max_epochs):
            self.model_.train()
            perm = torch.randperm(len(yt), generator=gen)
            total = 0.0
            for i in range(0, len(perm), cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                if len(idx) < 2:  # BatchNorm needs more than one sample
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model_(num_tr[idx], cat_tr[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            val_score = self._scores(num_val, cat_val)
            ap = average_precision_score(yv, val_score)
            sched.step(ap)
            self.history_.append({"epoch": epoch + 1, "train_loss": total / len(yt), "val_pr_auc": ap})
            if verbose:
                print(f"epoch {epoch + 1:3d}  loss={total / len(yt):.4f}  val PR-AUC={ap:.4f}")
            if ap > best + 1e-4:
                best, best_state, wait = ap, copy.deepcopy(self.model_.state_dict()), 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.fit_time_ = time.perf_counter() - t0
        self.best_val_pr_auc_ = best
        self.threshold_ = f2_threshold(yv, self._scores(num_val, cat_val))
        return self

    @torch.no_grad()
    def _scores(self, num, cat) -> np.ndarray:
        self.model_.eval()
        return torch.sigmoid(self.model_(num, cat)).numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        num, cat = map(torch.from_numpy, self.encoder_.transform(X))
        p = self._scores(num, cat)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @torch.no_grad()
    def embed(self, X: pd.DataFrame) -> np.ndarray:
        """Penultimate-layer representation, saved as a learned feature set."""
        self.model_.eval()
        num, cat = map(torch.from_numpy, self.encoder_.transform(X))
        return self.model_.represent(num, cat).numpy()


class TabNetWrapper:
    """pytorch-tabnet's TabNetClassifier on the same encoding, with the same F2 threshold policy."""

    def __init__(self, n_d=16, n_a=16, n_steps=4, gamma=1.5, max_epochs=60, patience=10, seed=config.SEED):
        self.params = dict(n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma)
        self.max_epochs, self.patience, self.seed = max_epochs, patience, seed

    def fit(self, X: pd.DataFrame, y: pd.Series):
        from pytorch_tabnet.tab_model import TabNetClassifier

        set_seed(self.seed)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.seed)
        self.encoder_ = TabularEncoder().fit(X_tr)
        A_tr, A_val = self._matrix(X_tr), self._matrix(X_val)
        n_num = self.encoder_.n_numeric_
        self.model_ = TabNetClassifier(
            **self.params,
            cat_idxs=list(range(n_num, n_num + len(self.encoder_.cardinalities_))),
            cat_dims=self.encoder_.cardinalities_,
            cat_emb_dim=4,
            optimizer_params={"lr": 2e-2},
            seed=self.seed,
            # GPU kernels are not bit-reproducible with the CPU ones, so a GPU runtime would train a
            # different model than the one reported; the table is small enough for the CPU.
            device_name="cpu",
            verbose=0,
        )
        t0 = time.perf_counter()
        self.model_.fit(
            A_tr,
            np.asarray(y_tr),
            eval_set=[(A_val, np.asarray(y_val))],
            eval_metric=["auc"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=1024,
            virtual_batch_size=128,
            weights=1,  # inverse-frequency sampling to counter the class imbalance
        )
        self.fit_time_ = time.perf_counter() - t0
        self.epochs_ = len(self.model_.history.history["loss"])
        self.threshold_ = f2_threshold(np.asarray(y_val), self.model_.predict_proba(A_val)[:, 1])
        return self

    def _matrix(self, X):
        num, cat = self.encoder_.transform(X)
        return np.hstack([num, cat.astype(np.float32)])

    def predict_proba(self, X):
        return self.model_.predict_proba(self._matrix(X))

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def feature_importances(self) -> pd.Series:
        names = self.encoder_.numeric_names_ + list(config.CATEGORICAL_FEATURES)
        return pd.Series(self.model_.feature_importances_, index=names).sort_values(ascending=False)


MLP_VARIANTS = {
    "MLP 128-64 (reference)": MLPConfig(),
    "MLP 64-32 (smaller)": MLPConfig(hidden=(64, 32)),
    "MLP 256-128-64 (deeper)": MLPConfig(hidden=(256, 128, 64)),
    "No dropout": MLPConfig(dropout=0.0),
    "One-hot instead of embeddings": MLPConfig(use_embeddings=False),
    "No class weighting": MLPConfig(pos_weight=False),
    "No engineered features": MLPConfig(feature_engineering=False),
}


def evaluate_network(name, model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    row = {"model": name, "fit_time": model.fit_time_, "threshold": model.threshold_}
    row.update(compute_metrics(y_test, (proba >= 0.5).astype(int), proba))
    row.update({f"{k}_tuned": v for k, v in compute_metrics(y_test, (proba >= model.threshold_).astype(int)).items()})
    return row


def summarize_seeds(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per model: the mean of every numeric column over the seeds plus ``<column>_std``
    (sample standard deviation across seeds)."""
    cols = runs.select_dtypes("number").columns.drop("seed")
    grouped = runs.groupby("model", sort=False)
    out = grouped[cols].mean().join(grouped[cols].std().add_suffix("_std")).reset_index()
    out.insert(1, "n_seeds", grouped.size().to_numpy())
    return out


def format_mean_std(summary: pd.DataFrame, cols, digits: int = 4) -> pd.DataFrame:
    """Display copy of a seed summary with ``mean ± sd`` strings."""
    out = summary[["model"]].copy()
    for c in cols:
        out[c] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(summary[c], summary[f"{c}_std"])]
    return out


def run_mlp_variants(X_train, y_train, X_test, y_test, variants=None, seeds=SEEDS, verbose=True):
    """Train each MLP variant once per seed; ``val_pr_auc`` (on the internal validation split) is
    what a practitioner would select on, the test columns are reported for completeness.

    Returns the per-variant summary (mean and sd over seeds), the per-seed runs and, per variant,
    the model trained with the first seed."""
    rows, models = [], {}
    for name, cfg in (variants or MLP_VARIANTS).items():
        for seed in seeds:
            m = MLPClassifier(cfg, seed=seed).fit(X_train, y_train)
            rows.append({**evaluate_network(name, m, X_test, y_test), "seed": seed,
                         "val_pr_auc": m.best_val_pr_auc_, "epochs": len(m.history_)})
            models.setdefault(name, m)
        if verbose:
            r = pd.DataFrame(rows[-len(seeds):])
            print(f"[MLP] {name:32s} val PR-AUC={r.val_pr_auc.mean():.4f}±{r.val_pr_auc.std():.4f} "
                  f"test PR-AUC={r.pr_auc.mean():.4f}±{r.pr_auc.std():.4f} "
                  f"({r.fit_time.mean():.0f}s, {r.epochs.mean():.0f} epochs per seed)")
    runs = pd.DataFrame(rows)
    summary = summarize_seeds(runs)
    config.ensure_dirs()
    runs.to_csv(config.RESULTS_DIR / "dl_mlp_variants_seeds.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "dl_mlp_variants.csv", index=False)
    return summary, runs, models


def run_tabnet(X_train, y_train, X_test, y_test, seeds=SEEDS, verbose=True):
    """Train TabNet once per seed; returns the summary row, the per-seed runs and the model
    trained with the first seed."""
    rows, models = [], []
    for seed in seeds:
        m = TabNetWrapper(seed=seed).fit(X_train, y_train)
        rows.append({**evaluate_network("TabNet", m, X_test, y_test), "seed": seed, "epochs": m.epochs_})
        models.append(m)
        if verbose:
            print(f"[TabNet] seed {seed}  test PR-AUC={rows[-1]['pr_auc']:.4f} ({m.fit_time_:.0f}s, {m.epochs_} epochs)")
    runs = pd.DataFrame(rows)
    config.ensure_dirs()
    runs.to_csv(config.RESULTS_DIR / "dl_tabnet_seeds.csv", index=False)
    return summarize_seeds(runs), runs, models[0]
