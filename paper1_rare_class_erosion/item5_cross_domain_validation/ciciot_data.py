"""
CICIoT2023 federated data loader for Item 5 (DESIGN_FROZEN.md). Loads the
pre-cleaned, pre-split parquet files produced by prepare_ciciot.py. Fits
the scaler and class weights on TRAIN only (Section 6/7 of the frozen
design). Dirichlet row-level partitioning (Section 7) happens on TRAIN
only, per (partition_seed, alpha, n_clients) -- validation/test never
touch federation at all.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

PREPARED_DIR = Path(__file__).resolve().parent / "prepared_data"


class CICIoTFederatedData:
    def __init__(self, prepared_dir=PREPARED_DIR):
        self.prepared_dir = prepared_dir
        with open(prepared_dir / "feature_columns.txt") as f:
            self.feature_cols = [l.strip() for l in f if l.strip()]
        with open(prepared_dir / "classes.txt") as f:
            self.all_categories = sorted(l.strip() for l in f if l.strip())
        self.label_encoder = LabelEncoder().fit(self.all_categories)

        self.train_df = pd.read_parquet(prepared_dir / "train.parquet")
        self.val_df = pd.read_parquet(prepared_dir / "validation.parquet")
        self.test_df = pd.read_parquet(prepared_dir / "test.parquet")

        self.scaler = None
        self.class_weights_full_balanced = None

    def fit_preprocessing(self):
        self.scaler = StandardScaler().fit(self.train_df[self.feature_cols])
        counts = self.train_df["cls"].value_counts().reindex(self.all_categories).fillna(0).values
        n, k = counts.sum(), len(self.all_categories)
        balanced = n / (k * np.maximum(counts, 1))
        self.class_weights_full_balanced = torch.tensor(balanced / balanced.mean(), dtype=torch.float32)
        print(f"Fitted scaler + {len(self.feature_cols)} feature columns on "
              f"{len(self.train_df)} TRAIN rows across {k} classes. Validation/test untouched.")

    def _to_tensors(self, df):
        X = self.scaler.transform(df[self.feature_cols])
        y = self.label_encoder.transform(df["cls"])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def get_validation_data(self):
        return self._to_tensors(self.val_df)

    def get_test_data(self):
        return self._to_tensors(self.test_df)

    def make_dirichlet_partition(self, partition_seed, alpha, n_clients):
        """Row-level synthetic label-skew Dirichlet partition of TRAIN only
        (DESIGN_FROZEN.md Section 7). For each class, draws p_c ~ Dir(alpha)
        over n_clients and assigns that class's TRAIN rows to clients
        according to p_c. Returns {client_id: (X, y)} and per-client row-count
        weights (summing to 1) for FedAvg-style aggregation weighting."""
        rng = np.random.RandomState(partition_seed)
        client_indices = {c: [] for c in range(n_clients)}
        rows_per_client_per_class = {cls: {c: 0 for c in range(n_clients)} for cls in self.all_categories}

        for cls in self.all_categories:
            idx = self.train_df.index[self.train_df["cls"] == cls].to_numpy()
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            p = rng.dirichlet([alpha] * n_clients)
            # proportional split via cumulative counts (deterministic given p and idx order)
            split_points = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            parts = np.split(idx, split_points)
            for c, part in zip(range(n_clients), parts):
                client_indices[c].extend(part.tolist())
                rows_per_client_per_class[cls][c] = len(part)

        client_data = {}
        weights = []
        for c in range(n_clients):
            sub = self.train_df.loc[client_indices[c]]
            client_data[c] = self._to_tensors(sub)
            weights.append(len(sub))
        total = sum(weights)
        weights = [w / total for w in weights]

        # holder_map[cls] = sorted list of client ids that received >=1 row of
        # that class in THIS partition draw (DESIGN_FROZEN.md Section 7 --
        # holder status is a per-draw consequence of the Dirichlet allocation,
        # not a fixed manifest as in ISOT).
        holder_map = {cls: sorted(c for c in range(n_clients) if rows_per_client_per_class[cls][c] > 0)
                      for cls in self.all_categories}
        return client_data, weights, holder_map

    def client_ids(self, n_clients):
        return list(range(n_clients))
