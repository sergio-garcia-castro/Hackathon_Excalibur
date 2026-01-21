"""
Training script with Leave-One-Patient-Out (LOPO) cross-validation
for a hierarchical Transformer:
  chunks -> session/day vector -> longitudinal embedding.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from models import (
    DayEmbeddingModel, 
    PositionalEmbedding, 
    DeltaDaysEmbedding, 
    SessionTransformer, 
    LinearClassifierHead
)

from utils import (
    compute_per_patient_auc,
    aggregate_patient_aucs,
    plot_embeddings_2d,
    print_evaluation_results,
    compute_random_baseline,
    extract_date_from_recording_id,
)
from config import get_config
from collections import defaultdict

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ----------------------------
# Session inference + time feats
# ----------------------------

def infer_sessions(df: pd.DataFrame,
                   patient_col="patient_short_id",
                   time_col="date",
                   gap_days=1):
    """
    Infer a session_id per patient based on large time gaps.
    Assumes time_col is numeric seconds OR pandas datetime.
    """
    df = df.copy()
    df[time_col] = df["recording_id"].apply(extract_date_from_recording_id)
    df = df.sort_values([patient_col, time_col])
  
    prev = df.groupby(patient_col)[time_col].shift(1)
    gap = df[time_col] - prev
    new_session = gap.isna() | (gap >= pd.Timedelta(days=gap_days))

    df["session_id"] = new_session.groupby(df[patient_col]).cumsum().astype(int)

    # session start time
    session_start = df.groupby([patient_col, "session_id"])[time_col].transform("min")
    df["session_start_time"] = session_start

    return df


def build_session_table(df: pd.DataFrame,
                        feature_cols=None,
                        patient_col="patient_short_id",
                        time_col="start_time",
                        gap_days=1):
    """
    Build a session-level index table for sequencing:
      one row per (patient, session_id),
      session_start_time, session_label (max),
      and the list of chunk indices belonging to the session.

    Returns:
      sessions_df: columns [patient_short_id, session_id, session_start_time, session_label, chunk_indices, #_chunks]
    """
    df = df.copy()
    df = infer_sessions(df, gap_days=gap_days)
    # Ensure sorted
    df = df.sort_values([patient_col, "session_start_time", time_col]).reset_index(drop=True)

    # Build chunk index lists per session
    grp = df.groupby([patient_col, "session_id"], sort=False)

    # Sessions grouping
    sessions = grp.agg(
        session_start_time=("session_start_time", "min"),
        session_label=("label", "max"),
    ).reset_index()

    # chunk indices for sampling
    # We use the dataframe row index as a stable pointer into features
    chunk_lists = grp.apply(lambda g: g.index.values).reset_index(name="chunk_indices")
    sessions = sessions.merge(chunk_lists, on=[patient_col, "session_id"], how="left")

    # session delta days per patient
    sessions = sessions.sort_values([patient_col, "session_start_time"]).reset_index(drop=True)

    prev = sessions.groupby(patient_col)["session_start_time"].shift(1)
    delta_days = (sessions["session_start_time"] - prev).dt.total_seconds() / 86400.0
    
    sessions["session_delta_days"] = delta_days.fillna(0.0).astype(np.float32)

    # a unique "recording id" at session level (for utils)
    # keep it stable and fold-safe
    sessions["session_rid"] = (
        sessions[patient_col].astype(str) + "_sess_" + sessions["session_id"].astype(str)
    )

    # Add total chunks
    sessions["total_chunks"] = sessions["chunk_indices"].apply(lambda x: len(x))


    return df, sessions

# ----------------------------
# Patient Timeline Dataset
# ----------------------------

class PatientTimelineDataset(Dataset):
    """
    One item = one patient's ordered sessions.
    Uses sessions_df['chunk_indices'] as pointers into chunks_df feature matrix.
    """
    def __init__(self, chunks_df, sessions_df, feature_cols,
                 patient_col="patient_short_id",
                 date_col="session_start_time",
                 label_col="session_label",
                 delta_col="session_delta_days",
                 rid_col="session_rid",
                 max_days=None):
        self.chunks_df = chunks_df.reset_index(drop=True)
        self.sessions_df = sessions_df.reset_index(drop=True)

        self.feature_cols = feature_cols
        self.patient_col = patient_col
        self.date_col = date_col
        self.label_col = label_col
        self.delta_col = delta_col
        self.rid_col = rid_col
        self.max_days = max_days

        # Prepack chunk features once for fast indexing
        self.X = self.chunks_df[self.feature_cols].to_numpy(dtype=np.float32)

        # Ensure sessions are ordered per patient by date
        self.sessions_df = self.sessions_df.sort_values([self.patient_col, self.date_col]).reset_index(drop=True)

        # Map patient -> list of session row indices
        by_patient = defaultdict(list)
        for i, row in self.sessions_df.iterrows():
            by_patient[row[self.patient_col]].append(i)

        self.patients = sorted(by_patient.keys())
        self.by_patient = by_patient

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        sess_rows = self.by_patient[patient_id]

        if self.max_days is not None:
            sess_rows = sess_rows[:self.max_days]

        sessions = []
        for r in sess_rows:
            row = self.sessions_df.iloc[r]
            sessions.append({
                "chunk_idx": torch.as_tensor(row["chunk_indices"], dtype=torch.long),
                "y": torch.tensor(row[self.label_col], dtype=torch.float32),
                "delta_days": torch.tensor(row[self.delta_col], dtype=torch.float32),
                "date": row[self.date_col],     # keep as datetime for debugging
                "rid": row[self.rid_col],
            })

        return {
            "patient_id": patient_id,
            "sessions": sessions,  # list length T_i
        }
    

def make_patient_timeline_collate_fn(X_np, max_days=None, max_chunks=None):
    """
    Returns a collate_fn that produces:
      x:          [B, T_max, K_max, F]
      chunk_mask: [B, T_max, K_max]  (True where real chunk)
      day_mask:   [B, T_max]         (True where real day)
      y:          [B, T_max]
      delta_days: [B, T_max]
      + metadata lists: patient_ids, dates, rids
    """
    X = torch.from_numpy(X_np)  # [N_chunks, F] float32 CPU tensor

    def collate(batch):
        B = len(batch)

        # Day dimension
        T_list = [len(b["sessions"]) for b in batch]
        T_max = max(T_list)
        if max_days is not None:
            T_max = min(T_max, max_days)

        # Chunk dimension (max across all included (patient, day))
        K_candidates = []
        for b in batch:
            for s in b["sessions"][:T_max]:
                K_candidates.append(len(s["chunk_idx"]))
        if len(K_candidates) == 0:
            raise ValueError("Empty batch: no sessions found.")
        K_max = max(K_candidates)
        if max_chunks is not None:
            K_max = min(K_max, max_chunks)

        F = X.shape[1]

        x = torch.zeros((B, T_max, K_max, F), dtype=torch.float32)
        chunk_mask = torch.zeros((B, T_max, K_max), dtype=torch.bool)
        day_mask = torch.zeros((B, T_max), dtype=torch.bool)

        y = torch.zeros((B, T_max), dtype=torch.float32)
        delta_days = torch.zeros((B, T_max), dtype=torch.float32)

        patient_ids = [b["patient_id"] for b in batch]
        dates = [[None] * T_max for _ in range(B)]
        rids  = [[None] * T_max for _ in range(B)]

        for i, b in enumerate(batch):
            sessions = b["sessions"][:T_max]
            Ti = len(sessions)
            day_mask[i, :Ti] = True

            for t, s in enumerate(sessions):
                idx = s["chunk_idx"]
                if max_chunks is not None:
                    idx = idx[:max_chunks]
                k = idx.numel()

                x[i, t, :k] = X.index_select(0, idx)   # gather [k, F]
                chunk_mask[i, t, :k] = True

                y[i, t] = s["y"]
                delta_days[i, t] = s["delta_days"]
                dates[i][t] = s["date"]
                rids[i][t]  = s["rid"]

        return {
            "x": x,  # [B,T,K,F]
            "chunk_mask": chunk_mask,  # [B,T,K]
            "day_mask": day_mask,      # [B,T]
            "y": y,                    # [B,T]
            "delta_days": delta_days,  # [B,T]
            "patient_ids": patient_ids,
            "dates": dates,
            "rids": rids,
        }

    return collate


# ----------------------------
# Training + evaluation loops
# ----------------------------

def train_epoch(day_model, pos_emb, delta_emb, emb_model, next_day_head, classifier, train_loader, criterion, optimizer, day_dim, device):
    day_model.train()
    pos_emb.train()
    delta_emb.train()
    emb_model.train()
    classifier.train()

    total_loss = 0.0
    n = 0

    for batch in train_loader:
        x = batch["x"].to(device)                    # [B,T,K,F]
        y = batch["y"].to(device)                    # [B,T]
        chunk_mask = batch["chunk_mask"].to(device)  # [B,T,K]
        day_mask = batch["day_mask"].to(device)      # [B,T] bool
        delta_days = batch["delta_days"].to(device).float()  # [B,T] float

        B, T, K, Ft = x.shape

        # --- Day embeddings ---
        x_flat = x.reshape(B*T, K, Ft)
        chunk_mask_flat = chunk_mask.reshape(B*T, K)

        day_emb_flat = day_model(x_flat, chunk_mask_flat)  # [B*T,D]
        day_emb = day_emb_flat.reshape(B, T, day_dim)      # [B,T,D]

        # --- Add time info ---
        pos = pos_emb(T, device=device)                    # [T,D]
        E = day_emb + pos.unsqueeze(0) + delta_emb(delta_days)     # [B,T,D]

        # --- zero padded days; not strictly necessary if masks are correct, but fine ---
        E = E * day_mask.unsqueeze(-1).type_as(E)

        # --- Temporal model (must be causal inside emb_model) ---
        H = emb_model(E, day_mask)                         # [B,T,D]  (NOT [B,D])

        # --- Next-token prediction: predict embedding for t+1 from state at t ---
        pred_next = next_day_head(H[:, :-1, :])            # [B,T-1,D]
        target_next = day_emb[:, 1:, :]                    # [B,T-1,D] (or E[:,1:,:] if you prefer)

        mask_next = day_mask[:, 1:]                        # valid targets (day t+1 exists)

        # --- Per-day label from predicted embedding (aligned to t+1) ---
        logits_next = classifier(pred_next)     # [B, T-1, 2]
        labels_next = y[:, 1:].long()                  # [B, T-1]
        mask_next   = day_mask[:, 1:]           # [B, T-1] bool (True = valid target)

        logits_flat = logits_next[mask_next]    # [N_valid, 2]
        labels_flat = labels_next[mask_next]    # [N_valid]

        loss = criterion(logits_flat, labels_flat)  # Loss

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_1(day_model, pos_emb, delta_emb, emb_model, next_day_head, classifier,
             loader, day_dim, device):
    day_model.eval()
    pos_emb.eval()
    delta_emb.eval()
    emb_model.eval()
    next_day_head.eval()
    classifier.eval()

    all_emb, all_pred, all_lab = [], [], []
    all_pids, all_rids = [], []

    for batch in loader:
        x = batch["x"].to(device)                    # [B,T,K,F]
        y = batch["y"].to(device)                    # [B,T]
        chunk_mask = batch["chunk_mask"].to(device)  # [B,T,K]
        day_mask = batch["day_mask"].to(device)      # [B,T] bool
        delta_days = batch["delta_days"].to(device).float()  # [B,T]

        # Optional IDs (recommended to include in your dataset)
        # pids: list/array length B
        pids = batch.get("patient_ids", None)
        # day ids/dates: [B,T] (strings, ints, datetimes, etc.) or None
        day_ids = batch.get("day_id", batch.get("dates", None))

        B, T, K, Ft = x.shape

        # --- Day embeddings ---
        x_flat = x.reshape(B * T, K, Ft)
        chunk_mask_flat = chunk_mask.reshape(B * T, K)

        day_emb_flat = day_model(x_flat, chunk_mask_flat)  # [B*T,D]
        day_emb = day_emb_flat.reshape(B, T, day_dim)      # [B,T,D]

        # --- Add time info ---
        pos = pos_emb(T, device=device)                    # [T,D]
        E = day_emb + pos.unsqueeze(0) + delta_emb(delta_days)  # [B,T,D]
        E = E * day_mask.unsqueeze(-1).type_as(E)

        # --- Temporal model ---
        H = emb_model(E, day_mask)                         # [B,T,D]

        # --- Next-token prediction (t -> t+1) ---
        pred_next = next_day_head(H[:, :-1, :])            # [B,T-1,D]
        logits_next = classifier(pred_next)                # [B,T-1,2]
        probs_next = torch.softmax(logits_next, dim=-1)[..., 1]  # [B,T-1]

        labels_next = y[:, 1:].long()                      # [B,T-1]
        mask_next = day_mask[:, 1:]                        # [B,T-1] bool

        # --- Select only valid target days ---
        pred_next_valid = pred_next[mask_next]             # [N_valid, D]
        probs_valid = probs_next[mask_next]                # [N_valid]
        labels_valid = labels_next[mask_next]              # [N_valid]

        all_emb.append(pred_next_valid.detach().cpu().numpy())
        all_pred.append(probs_valid.detach().cpu().numpy())
        all_lab.append(labels_valid.detach().cpu().numpy())

        # --- IDs aligned to valid predictions ---
        # patient id repeated per valid target day
        if pids is None:
            # fallback: integer patient index within this batch
            pids_arr = np.arange(B)
        else:
            pids_arr = np.asarray(pids)

        # Build per-position patient ids [B, T-1] then mask
        pids_grid = np.repeat(pids_arr[:, None], T - 1, axis=1)   # [B,T-1]
        all_pids.extend(pids_grid[mask_next.detach().cpu().numpy()].tolist())

        # day ids aligned to t+1 (targets)
        if day_ids is None:
            # fallback: use (t+1) index within sequence
            rid_grid = np.tile(np.arange(1, T), (B, 1))           # [B,T-1], values 1..T-1
        else:
            day_ids_arr = np.asarray(day_ids)                    # expect [B,T]
            rid_grid = day_ids_arr[:, 1:]                        # [B,T-1]

        all_rids.extend(rid_grid[mask_next.detach().cpu().numpy()].tolist())

    # Handle empty case safely
    if len(all_emb) == 0:
        return (
            np.zeros((0, day_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=object),
        )

    return (
        np.vstack(all_emb),                # [N_valid, D] predicted next-day embeddings
        np.concatenate(all_pred),          # [N_valid] P(class=1)
        np.concatenate(all_lab),           # [N_valid] true label (0/1)
        np.array(all_pids, dtype=object),  # [N_valid] patient ids
        np.array(all_rids, dtype=object),  # [N_valid] day ids (or indices)
    )

@torch.no_grad()
def evaluate(
    day_model, pos_emb, delta_emb, emb_model, next_day_head, classifier,
    loader, day_dim, device
):
    day_model.eval()
    pos_emb.eval()
    delta_emb.eval()
    emb_model.eval()
    next_day_head.eval()
    classifier.eval()

    # Content-only day embeddings (from day_model)
    all_D_prev, all_D_next = [], []

    # Content+time embeddings that feed the transformer
    all_E_prev, all_E_next = [], []

    # Transformer state and predicted-next embeddings
    all_H_next, all_P_next = [], []

    all_prob, all_lab = [], []
    all_pids, all_rids = [], []

    for batch in loader:
        x = batch["x"].to(device)                    # [B,T,K,F]
        y = batch["y"].to(device)                    # [B,T]
        chunk_mask = batch["chunk_mask"].to(device)  # [B,T,K]
        day_mask = batch["day_mask"].to(device)      # [B,T] bool/0-1
        delta_days = batch["delta_days"].to(device).float()  # [B,T]

        pids = batch.get("patient_ids", None)        # len B (optional)
        day_ids = batch.get("day_id", batch.get("dates", batch.get("rids", None)))  # [B,T] (optional)

        B, T, K, Ft = x.shape

        # ---- Day embeddings (content-only): D [B,T,D_out] ----
        x_flat = x.reshape(B * T, K, Ft)
        cm_flat = chunk_mask.reshape(B * T, K)
        day_out = day_model(x_flat, cm_flat)         # [B*T, D_out]
        D_out = day_out.shape[-1]
        D = day_out.reshape(B, T, D_out)             # [B,T,D_out]

        # ---- Add time info to form transformer input: E [B,T,D_out] ----
        pos = pos_emb(T, device=device)              # [T,D_out]
        E = D + pos.unsqueeze(0) + delta_emb(delta_days)  # [B,T,D_out]
        E = E * day_mask.unsqueeze(-1).type_as(E)

        # ---- Transformer states: H [B,T,D_out] ----
        H = emb_model(E, day_mask)

        # ---- Predicted next embedding from past: P_next [B,T-1,D_out] ----
        P_next = next_day_head(H[:, :-1, :])         # uses time t to predict t+1
        logits = classifier(P_next)                  # [B,T-1,2]
        probs = torch.softmax(logits, dim=-1)[..., 1]  # [B,T-1]

        # ---- Alignment to targets (t+1) ----
        # Targets correspond to indices 1..T-1
        mask_next = day_mask[:, 1:]                  # [B,T-1] valid targets
        labels_next = y[:, 1:].long()                # [B,T-1]

        # For any "*_next" we take [:,1:,:] and apply mask_next
        D_next = D[:, 1:, :]                         # [B,T-1,D_out]
        E_next = E[:, 1:, :]                         # [B,T-1,D_out]
        H_next = H[:, 1:, :]                         # [B,T-1,D_out]

        # For any "*_prev" we take [:,:-1,:] but apply the SAME mask_next
        # so each row is the "previous day representation" for a valid target day.
        D_prev = D[:, :-1, :]                        # [B,T-1,D_out]
        E_prev = E[:, :-1, :]                        # [B,T-1,D_out]

        # ---- Select only valid target positions ----
        mask_np = mask_next.detach().cpu().numpy()

        all_D_prev.append(D_prev[mask_next].detach().cpu().numpy())
        all_D_next.append(D_next[mask_next].detach().cpu().numpy())
        all_E_prev.append(E_prev[mask_next].detach().cpu().numpy())
        all_E_next.append(E_next[mask_next].detach().cpu().numpy())
        all_H_next.append(H_next[mask_next].detach().cpu().numpy())
        all_P_next.append(P_next[mask_next].detach().cpu().numpy())

        all_prob.append(probs[mask_next].detach().cpu().numpy())
        all_lab.append(labels_next[mask_next].detach().cpu().numpy())

        # ---- IDs aligned to valid targets (t+1) ----
        if pids is None:
            pids_arr = np.arange(B)
        else:
            pids_arr = np.asarray(pids)

        pids_grid = np.repeat(pids_arr[:, None], T - 1, axis=1)  # [B,T-1]
        all_pids.extend(pids_grid[mask_np].tolist())

        if day_ids is None:
            rid_grid = np.tile(np.arange(1, T), (B, 1))          # [B,T-1] -> 1..T-1
        else:
            day_ids_arr = np.asarray(day_ids, dtype=object)      # [B,T]
            rid_grid = day_ids_arr[:, 1:]                        # [B,T-1]

        all_rids.extend(rid_grid[mask_np].tolist())

    # ---- Empty case ----
    if len(all_lab) == 0:
        Z = np.zeros((0, day_dim), dtype=np.float32)
        return (
            Z, Z, Z, Z, Z, Z,                        # D_prev, D_next, E_prev, E_next, H_next, P_next
            np.zeros((0,), dtype=np.float32),        # probs
            np.zeros((0,), dtype=np.int64),          # labels
            np.zeros((0,), dtype=object),            # pids
            np.zeros((0,), dtype=object),            # rids
        )

    return (
        np.vstack(all_D_prev),                       # [N_valid, D] content-only day_emb at t
        np.vstack(all_D_next),                       # [N_valid, D] content-only day_emb at t+1
        np.vstack(all_E_prev),                       # [N_valid, D] transformer input embedding at t (content+time)
        np.vstack(all_E_next),                       # [N_valid, D] transformer input embedding at t+1 (content+time)
        np.vstack(all_H_next),                       # [N_valid, D] transformer hidden state at t+1
        np.vstack(all_P_next),                       # [N_valid, D] predicted-next embedding (from t -> t+1)
        np.concatenate(all_prob),                    # [N_valid] P(class=1)
        np.concatenate(all_lab),                     # [N_valid] y at t+1
        np.array(all_pids, dtype=object),            # [N_valid]
        np.array(all_rids, dtype=object),            # [N_valid]
    )


# ----------------------------
# Data loading
# ----------------------------

def load_df(data_path: str):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # minimal required cols
    required = ["recording_id", "patient_short_id", "label", "start_time", "end_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} chunks from {df['patient_short_id'].nunique()} patients")

    # Determine feature columns (exclude known metadata)
    metadata_cols = [
        "recording_id", "patient_short_id", "label",
        "start_time", "end_time",
        "age", "sex", "audio_quality",
    ]
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    # Clean NaNs in features
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Add date to df
    df["date"] = df["recording_id"].apply(extract_date_from_recording_id)

    return df, feature_cols


# ----------------------------
# LOPO training
# ----------------------------

def train_lopo(
    df: pd.DataFrame,
    feature_cols,
    *,
    chunk_hidden: int = 128,
    day_dim: int = 128,
    gap_days: int = 1,
    max_T_emb: int = 50,
    dropout: float = 0.3,
    batch_size: int = 32,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.,
    device: str = "cpu",
    save_path: str = "../emb_vis/"
):
    unique_patients = df["patient_short_id"].unique()
    print(f"\nStarting LOPO with {len(unique_patients)} folds...\n")

    all_per_patient_aucs = {}

    # Store all embedding spaces across folds
    all_D_prev, all_D_next = [], []
    all_E_prev, all_E_next = [], []
    all_H_next, all_P_next = [], []

    # Store common evaluation outputs across folds
    all_test_labels = []
    all_test_patient_ids = []
    all_test_recording_ids = []
    all_test_probs = []

    for test_patient in unique_patients:
        print(f"\n{'=' * 60}")
        print(f"Fold: Holding out {test_patient}")
        print(f"{'=' * 60}")

        train_df = df[df["patient_short_id"] != test_patient].copy()
        test_df = df[df["patient_short_id"] == test_patient].copy()

        # Fit scaler on TRAIN CHUNKS ONLY
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
        X_test  = scaler.transform(test_df[feature_cols].values.astype(np.float32))

        # Patient-centering AFTER scaling
        def patient_center(X, pids):
            Xc = X.copy()
            pids = np.asarray(pids)
            for pid in np.unique(pids):
                m = (pids == pid)
                mu = Xc[m].mean(axis=0, keepdims=True)
                Xc[m] = Xc[m] - mu
            return Xc

        X_train = patient_center(X_train, train_df["patient_short_id"].values)
        X_test  = patient_center(X_test,  test_df["patient_short_id"].values)

        # Write back into dfs so chunk_indices remain valid
        train_df_scaled = train_df.copy()
        test_df_scaled  = test_df.copy()
        train_df_scaled[feature_cols] = X_train
        test_df_scaled[feature_cols]  = X_test

        # Build session tables
        train_df2, train_sessions = build_session_table(train_df_scaled, feature_cols, gap_days=gap_days)
        test_df2,  test_sessions  = build_session_table(test_df_scaled,  feature_cols, gap_days=gap_days)

        print(f"Train sessions: {len(train_sessions)} | Test sessions: {len(test_sessions)}")
        print("Train label dist (session):", np.bincount(train_sessions["session_label"].astype(int).values, minlength=2))
        print("Test label dist (session):",  np.bincount(test_sessions["session_label"].astype(int).values, minlength=2))

        # Datasets
        train_dataset = PatientTimelineDataset(train_df2, train_sessions, feature_cols)
        test_dataset  = PatientTimelineDataset(test_df2,  test_sessions,  feature_cols)

        train_collate_fn = make_patient_timeline_collate_fn(train_dataset.X, max_days=None, max_chunks=None)
        test_collate_fn  = make_patient_timeline_collate_fn(test_dataset.X,  max_days=None, max_chunks=None)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=train_collate_fn, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=test_collate_fn, num_workers=0
        )

        # Model
        input_dim = len(feature_cols)
        day_model = DayEmbeddingModel(input_dim, chunk_hidden, day_dim, dropout=dropout).to(device)

        pos_emb = PositionalEmbedding(day_dim, max_len=max_T_emb).to(device)
        delta_emb = DeltaDaysEmbedding(day_dim).to(device)
        emb_model = SessionTransformer(day_dim, n_heads=4, n_layers=2, d_ff=None, dropout=dropout).to(device)

        next_day_head = nn.Linear(day_dim, day_dim).to(device)
        classifier = LinearClassifierHead(embedding_dim=day_dim, num_classes=2).to(device)

        def count_trainable_params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        modules = {
            "day_model": day_model,
            "pos_emb": pos_emb,
            "delta_emb": delta_emb,
            "emb_model": emb_model,
            "next_day_head": next_day_head,
            "classifier": classifier,
        }

        total_params = 0
        print("\nTrainable parameters per module:")
        for name, m in modules.items():
            n_params = count_trainable_params(m)
            total_params += n_params
            print(f"  {name:<14}: {n_params:,}")
        print(f"  {'TOTAL PARAMS':<14}: {total_params:,}\n")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(day_model.parameters()) +
            list(pos_emb.parameters()) +
            list(delta_emb.parameters()) +
            list(emb_model.parameters()) +
            list(next_day_head.parameters()) +
            list(classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        best_loss = float("inf")
        for epoch in range(num_epochs):
            loss = train_epoch(
                day_model, pos_emb, delta_emb, emb_model, next_day_head, classifier,
                train_loader, criterion, optimizer, day_dim, device
            )
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}")
            best_loss = min(best_loss, loss)

        # ---- Evaluate ----
        (
            test_D_prev, test_D_next,
            test_E_prev, test_E_next,
            test_H_next, test_P_next,
            test_probs, test_y,
            test_pids, test_rids
        ) = evaluate(
            day_model, pos_emb, delta_emb, emb_model, next_day_head, classifier,
            test_loader, day_dim, device
        )

        # Store embeddings
        all_D_prev.append(test_D_prev); all_D_next.append(test_D_next)
        all_E_prev.append(test_E_prev); all_E_next.append(test_E_next)
        all_H_next.append(test_H_next); all_P_next.append(test_P_next)

        # Store eval outputs
        all_test_probs.append(test_probs)
        all_test_labels.append(test_y)
        all_test_patient_ids.append(test_pids)
        all_test_recording_ids.append(test_rids)

        # Compute per-patient AUC at SESSION level (session_rid)
        per_patient_auc = compute_per_patient_auc(test_pids, test_y, test_probs, test_rids)
        all_per_patient_aucs.update(per_patient_auc)

        for pid, auc in per_patient_auc.items():
            if auc is not None:
                print(f"\n{test_patient} ROC AUC (session level): {auc:.4f}")

    # Aggregate results
    mean_auc, std_auc, valid_aucs = aggregate_patient_aucs(all_per_patient_aucs)

    # Random baseline
    all_test_recording_ids_array = np.concatenate(all_test_recording_ids)
    random_mean, random_std, _ = compute_random_baseline(
        np.concatenate(all_test_patient_ids),
        np.concatenate(all_test_labels),
        all_test_recording_ids_array,
        n_iterations=100,
    )

    print_evaluation_results(all_per_patient_aucs, mean_auc, std_auc, "LOPO", (random_mean, random_std))

    # ---- Viz: stack everything across folds ----
    D_prev = np.vstack(all_D_prev) if len(all_D_prev) else np.zeros((0, day_dim), np.float32)
    D_next = np.vstack(all_D_next) if len(all_D_next) else np.zeros((0, day_dim), np.float32)
    E_prev = np.vstack(all_E_prev) if len(all_E_prev) else np.zeros((0, day_dim), np.float32)
    E_next = np.vstack(all_E_next) if len(all_E_next) else np.zeros((0, day_dim), np.float32)
    H_next = np.vstack(all_H_next) if len(all_H_next) else np.zeros((0, day_dim), np.float32)
    P_next = np.vstack(all_P_next) if len(all_P_next) else np.zeros((0, day_dim), np.float32)

    labels = np.concatenate(all_test_labels) if len(all_test_labels) else np.zeros((0,), np.int64)
    patient_ids = np.concatenate(all_test_patient_ids) if len(all_test_patient_ids) else np.zeros((0,), dtype=object)

    # You can decide which one is your "main" space; usually P_next is the clean predictive embedding.
    plot_embeddings_2d(
        P_next,
        labels,
        patient_ids,
        title=f"P_next (predict t→t+1) Embeddings (LOPO) - Mean AUC: {mean_auc:.4f}",
        save_path=f"{save_path}embeddings_P_next.png",
    )
    plt.show()

    # Additional plots (optional but requested)
    plot_embeddings_2d(
        D_prev, labels, patient_ids,
        title="D_prev (content-only day_emb at t) - aligned to target (t+1)",
        save_path=f"{save_path}embeddings_D_prev.png",
    )
    plt.show()

    plot_embeddings_2d(
        D_next, labels, patient_ids,
        title="D_next (content-only day_emb at t+1) - target day content",
        save_path=f"{save_path}embeddings_D_next.png",
    )
    plt.show()

    plot_embeddings_2d(
        E_prev, labels, patient_ids,
        title="E_prev (content+time input at t) - aligned to target (t+1)",
        save_path=f"{save_path}embeddings_E_prev.png",
    )
    plt.show()

    plot_embeddings_2d(
        E_next, labels, patient_ids,
        title="E_next (content+time input at t+1) - target day input",
        save_path=f"{save_path}embeddings_E_next.png",
    )
    plt.show()

    plot_embeddings_2d(
        H_next, labels, patient_ids,
        title="H_next (transformer state at t+1) - beware non-causal leakage",
        save_path=f"{save_path}embeddings_H_next.png",
    )
    plt.show()

    return {
        "per_patient_auc": all_per_patient_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,

        # return all embeddings 
        "D_prev": D_prev,
        "D_next": D_next,
        "E_prev": E_prev,
        "E_next": E_next,
        "H_next": H_next,
        "P_next": P_next,

        "labels": labels,
        "patient_ids": patient_ids,
    }


def main():
    config = get_config()
    config.print_config()

    DEVICE = "cuda" if (torch.cuda.is_available() and config.use_cuda) else "cpu"
    print(f"Using device: {DEVICE}")

    df, feature_cols = load_df(config.data_path)

    # You’ll likely want to move S,C,stride,gap_days into config.py
    results = train_lopo(
        df=df,
        feature_cols=feature_cols,
        chunk_hidden=config.chunk_hidden,
        day_dim=config.day_dim,
        gap_days=config.gap_days,
        max_T_emb=config.max_T_emb,
        dropout=config.dropout,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=DEVICE,
    )

    print("\nTraining complete!")
    print(f"Final Mean ROC AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    print("Embedding visualization saved to: embeddings_visualization.png")


if __name__ == "__main__":
    main()
