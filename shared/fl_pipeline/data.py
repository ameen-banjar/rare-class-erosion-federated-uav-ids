"""
Data loading for the FL baseline pipeline. Reads the FROZEN
federated_clients_manifest.json (../data_prep/) -- this module never
regenerates it. Held-out test sessions are only ever touched by
evaluate_final_test(), called once per run at the very end.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from prepare_isot import DATA_DIR, TIMESTAMP_LEAK_COLUMNS  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parent.parent / "data_prep"
MANIFEST_PATH = AUDIT_DIR / "federated_clients_manifest.json"
ALL_CATEGORIES = ["DoS", "Injection", "Ip Spoofing", "MITM", "Manipulation",
                   "Password Cracking", "Regular", "Replay", "Unauth", "Video"]

_session_cache = {}


def _load_session(session_name, category):
    key = (session_name, category)
    if key in _session_cache:
        return _session_cache[key]
    f = DATA_DIR / category / f"{session_name}.csv"
    d = pd.read_csv(f)
    d = d.drop(columns=[c for c in TIMESTAMP_LEAK_COLUMNS if c in d.columns])
    d["_category"] = category
    _session_cache[key] = d
    return d


class ISOTFederatedData:
    def __init__(self, manifest_path=MANIFEST_PATH):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.n_clients = self.manifest["n_clients"]
        self._resolve_categories()

        self.feature_cols = None  # set on first load
        self.scaler = None
        self.label_encoder = LabelEncoder().fit(ALL_CATEGORIES)

    def _resolve_categories(self):
        # session name -> category, by scanning the dataset directory once (137 files, cheap)
        self._cat_of = {}
        for cat_dir in DATA_DIR.iterdir():
            if cat_dir.is_dir():
                for f in cat_dir.glob("*.csv"):
                    self._cat_of[f.stem] = cat_dir.name

    def _load_sessions(self, session_names):
        frames = [_load_session(s, self._cat_of[s]) for s in session_names]
        return pd.concat(frames, ignore_index=True)

    def fit_preprocessing(self):
        """Fit scaler + class weights from client TRAINING sessions only --
        NOT validation, NOT held_out_test. Validation must stay an unbiased
        proxy for generalization; including it in preprocessing (as an
        earlier version of this function did) contaminates that independence
        even though it never touches the final test set."""
        pool_sessions = []
        for c in self.manifest["clients"].values():
            pool_sessions += c["sessions"]
        df = self._load_sessions(pool_sessions)

        non_feature = {"_category"}
        cols = [c for c in df.columns if c not in non_feature]
        numeric = df[cols].select_dtypes(include=[np.number]).columns.tolist()
        still_bad = [c for c in numeric if df[c].abs().median() > 1e8]
        if still_bad:
            raise RuntimeError(f"Epoch-scale columns leaked through: {still_bad}")
        self.feature_cols = numeric

        self.scaler = StandardScaler().fit(df[self.feature_cols].fillna(0))

        # class weights from the pool label distribution only (never test).
        # Regular/DoS dominate ~85% of pool rows while some attack families
        # have <10k rows -- plain unweighted CrossEntropyLoss collapses to
        # predicting only the majority classes (observed empirically: 7/10
        # classes at 0.0 recall in early rounds). Balanced weight, sqrt-damped
        # to avoid extreme gradients from the ~270x frequency ratio, mean-
        # normalized to 1 so the overall loss scale is unchanged.
        counts = df["_category"].value_counts().reindex(ALL_CATEGORIES).fillna(0).values
        n, k = counts.sum(), len(ALL_CATEGORIES)
        balanced = n / (k * np.maximum(counts, 1))
        damped = np.sqrt(balanced)
        self.class_weights = torch.tensor(damped / damped.mean(), dtype=torch.float32)
        self.class_weights_full_balanced = torch.tensor(balanced / balanced.mean(), dtype=torch.float32)

        def effective_share(w):
            eff = counts * w.numpy()
            return {c: float(s) for c, s in zip(ALL_CATEGORIES, eff / eff.sum())}

        print(f"Fitted scaler + {len(self.feature_cols)} feature columns on "
              f"{len(pool_sessions)} TRAINING-pool sessions ({len(df)} rows). Validation and test untouched.")
        print("Class weights (sqrt-damped, mean-normalized):",
              {c: round(float(w), 3) for c, w in zip(ALL_CATEGORIES, self.class_weights)})
        print("  -> effective row share after sqrt-damped weighting:",
              {c: round(v, 4) for c, v in effective_share(self.class_weights).items()})
        print("  -> effective row share after FULL balanced weighting:",
              {c: round(v, 4) for c, v in effective_share(self.class_weights_full_balanced).items()})

    def _to_tensors(self, df):
        X = self.scaler.transform(df[self.feature_cols].fillna(0))
        y = self.label_encoder.transform(df["_category"])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def get_client_data(self, client_id):
        sessions = self.manifest["clients"][str(client_id)]["sessions"]
        df = self._load_sessions(sessions)
        return self._to_tensors(df)

    def get_client_raw_df(self, client_id):
        """Raw (unscaled) feature dataframe + category column, for the drift
        module to perturb BEFORE scaling. Use tensors_from_df() afterward."""
        sessions = self.manifest["clients"][str(client_id)]["sessions"]
        return self._load_sessions(sessions)

    def tensors_from_df(self, df):
        return self._to_tensors(df)

    def get_all_client_data_pooled(self):
        """For the centralized-oracle baseline only."""
        all_sessions = []
        for c in self.manifest["clients"].values():
            all_sessions += c["sessions"]
        df = self._load_sessions(all_sessions)
        return self._to_tensors(df)

    def get_validation_data(self):
        df = self._load_sessions(self.manifest["validation_sessions"])
        return self._to_tensors(df)

    def get_final_test_data(self):
        """Touch this only once, at the very end, per the frozen-manifest protocol."""
        df = self._load_sessions(self.manifest["held_out_test_sessions"])
        return self._to_tensors(df)

    def client_weight_rows(self, client_id):
        return self.manifest["clients"][str(client_id)]["fedavg_weight_by_rows"]

    def client_ids(self):
        return list(range(self.n_clients))
