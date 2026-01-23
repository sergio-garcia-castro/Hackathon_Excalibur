from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score

import json
from dataclasses import asdict


import math
from pathlib import Path
import pickle

# 0) NOTATION
# Segment = one parquet row (acoustic feature vector + metadata)
# Session = one (patient_short_id, recording_id), label constant within session
# We build training examples at the session level,
#   For each session j, we keep up to M segments (tokens), ordered by start_time.
#   For each patient, we also build a causal window of K sessions ending at session j.

# Input tensors per batch:
#   X_win   : (B, K, M, D)   residualized acoustic features
#   pos_win : (B, K, M)      within-session normalized position in [0,1]
#   dt_win  : (B, K, M)      within-session time gaps (seconds)
#   gap_win : (B, K)         session gaps (days) to previous session (patient-relative)
#   len_win : (B, K)         valid segment counts per session token-list
# Output:
#   z       : (B, H)         embedding for CURRENT session (last in window)
#   logits  : (B, 2)         linear head (repo constraint)


# 1) CONFIG
@dataclass
class CFG:
    data_path: str = "data/dataset.parquet"  # Path is relative to parent dir
    random_seed: int = 0

    # folds: LOPO over patients
    use_cuda: bool = True

    # scaling
    scaler_type: str = "robust"  # "robust" or "standard"
    clip_scaled: float = 0.0  # e.g. 5.0, 0 disables

    # causal per-patient baseline removal (feature space)
    ema_alpha: float = 0.95
    ema_eps: float = 1e-5

    # tokenization from segments
    max_segments_per_session: int = 512  # M
    session_window: int = 4  # K
    segment_sampling: str = "uniform"  # "uniform" | "random"

    # model sizes
    d_model: int = 64
    seg_n_layers: int = 2
    sess_n_layers: int = 2
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1

    # relative time encodings
    n_time_freqs: int = 4
    use_pos: bool = True
    use_dt: bool = False
    use_session_gaps: bool = True

    # training
    batch_size: int = 8
    num_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0


# 2) TIME PARSING (recording_id -> session date)
_DATE_RE = re.compile(r"(19|20)\d{2}-\d{2}-\d{2}")


def parse_session_time_seconds(recording_ids: np.ndarray) -> np.ndarray:
    """
    Parse a session timestamp from recording_id.
    Expected pattern: ".../YYYY-MM-DD".
    """
    rids = recording_ids.astype(str)
    dates = []
    for s in rids:
        m = _DATE_RE.search(s)
        dates.append(m.group(0) if m else None)

    ser = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True)
    if ser.notna().mean() > 0.9:
        return (ser.astype("int64").to_numpy() / 1e9).astype(np.float64)

    ranks, _ = pd.factorize(rids, sort=True)
    return ranks.astype(np.float64)


def make_scaler(scaler_type: str):
    if scaler_type == "robust":
        return RobustScaler(
            with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
        )
    if scaler_type == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler_type={scaler_type}")


def causal_patient_residualize_by_session(
    X: np.ndarray,
    patient_ids: np.ndarray,
    recording_ids: np.ndarray,
    start_time: Optional[np.ndarray],
    alpha: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= alpha < 1.0):
        raise ValueError("alpha must be in [0,1)")

    n, d = X.shape
    Xr = np.zeros_like(X, dtype=np.float32)
    gap_days = np.zeros((n,), dtype=np.float32)

    sess_t = parse_session_time_seconds(recording_ids)
    seg_key = (
        np.arange(n, dtype=np.float64)
        if start_time is None
        else start_time.astype(np.float64)
    )

    for pid in np.unique(patient_ids):
        idxs_p = np.where(patient_ids == pid)[0]
        if idxs_p.size == 0:
            continue

        rids_p = recording_ids[idxs_p].astype(str)
        sess_t_p = sess_t[idxs_p]

        uniq_rids, inv = np.unique(rids_p, return_inverse=True)
        rid_time = {
            rid: float(np.median(sess_t_p[inv == k])) for k, rid in enumerate(uniq_rids)
        }
        uniq_rids_sorted = sorted(uniq_rids, key=lambda rid: rid_time[rid])

        # patient-relative session gaps (days)
        prev_t = None
        rid_gap = {}
        for rid in uniq_rids_sorted:
            t = rid_time[rid]
            rid_gap[rid] = 0.0 if prev_t is None else max((t - prev_t) / 86400.0, 0.0)
            prev_t = t

        mu = np.zeros((d,), dtype=np.float32)
        m2 = np.zeros((d,), dtype=np.float32)
        var = np.ones((d,), dtype=np.float32)

        for rid in uniq_rids_sorted:
            idxs_sess = idxs_p[rids_p == rid]
            o = np.argsort(seg_key[idxs_sess], kind="mergesort")
            idxs_sess = idxs_sess[o]

            denom = np.sqrt(var + eps)
            Xr[idxs_sess] = (X[idxs_sess].astype(np.float32) - mu[None, :]) / denom[
                None, :
            ]
            gap_days[idxs_sess] = float(rid_gap[rid])

            # update baseline AFTER residualizing this session (strictly causal)
            for idx in idxs_sess:
                x = X[idx].astype(np.float32)
                mu = alpha * mu + (1.0 - alpha) * x
                m2 = alpha * m2 + (1.0 - alpha) * (x * x)
                var = np.maximum(m2 - mu * mu, eps).astype(np.float32)

    return Xr, gap_days


# 4) SESSION TOKENIZATION
def build_sessions(
    X: np.ndarray,
    y: np.ndarray,
    pid: np.ndarray,
    rid: np.ndarray,
    start_time: Optional[np.ndarray],
    end_time: Optional[np.ndarray],
    gap_days_per_segment: Optional[np.ndarray],
    max_segments: int,
    sampling: str,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    Build per-session token arrays
    Each session stores up to M segment tokens.
    """
    df = pd.DataFrame(
        {
            "pid": pid,
            "rid": rid.astype(str),
            "y": y.astype(np.int64),
            "_i": np.arange(len(y), dtype=np.int64),
            "s": (start_time.astype(np.float32) if start_time is not None else 0.0),
            "e": (end_time.astype(np.float32) if end_time is not None else 1.0),
        }
    )
    gb = df.groupby(["pid", "rid"], sort=False)

    R = gb.ngroups
    M = int(max_segments)
    D = int(X.shape[1])

    X_sess = np.zeros((R, M, D), dtype=np.float32)
    pos_sess = np.zeros((R, M), dtype=np.float32)
    dt_sess = np.zeros((R, M), dtype=np.float32)
    len_sess = np.zeros((R,), dtype=np.int64)
    y_sess = np.zeros((R,), dtype=np.int64)
    pid_sess = np.empty((R,), dtype=object)
    rid_sess = np.empty((R,), dtype=object)

    # session time for ordering (from recording_id)
    sess_time = np.zeros((R,), dtype=np.float64)

    for k, ((p, r), g) in enumerate(gb):
        idxs = g["_i"].to_numpy()
        s = g["s"].to_numpy()
        e = g["e"].to_numpy()

        # sort segments by start_time
        o = np.argsort(s, kind="mergesort")
        idxs, s, e = idxs[o], s[o], e[o]
        T = len(idxs)
        if T == 0:
            continue

        if sampling == "uniform":
            if T > M:
                sel = np.round(np.linspace(0, T - 1, M)).astype(np.int64)
                idx_sel, s_sel, e_sel = idxs[sel], s[sel], e[sel]
            else:
                idx_sel, s_sel, e_sel = idxs, s, e
        elif sampling == "random":
            sel = rng.integers(0, T, size=M)
            sel.sort()
            idx_sel, s_sel, e_sel = idxs[sel], s[sel], e[sel]
        else:
            raise ValueError("sampling must be 'uniform' or 'random'")

        L = min(len(idx_sel), M)
        X_tok = X[idx_sel[:L]].astype(np.float32)

        # within-session normalized position in [0,1]
        s0 = float(np.min(s)) if len(s) else 0.0
        e1 = float(np.max(e)) if len(e) else (s0 + 1.0)
        dur = max(e1 - s0, 1.0)
        pos = ((s_sel[:L] - s0) / dur).astype(np.float32)

        # within-session dt between consecutive sampled segments (seconds if s is ms)
        dt = np.zeros((L,), dtype=np.float32)
        if L > 1:
            dt[1:] = np.maximum((s_sel[1:L] - s_sel[: L - 1]) / 1000.0, 0.0).astype(
                np.float32
            )

        X_sess[k, :L] = X_tok
        pos_sess[k, :L] = pos
        dt_sess[k, :L] = dt
        len_sess[k] = L
        y_sess[k] = int(g["y"].max())
        pid_sess[k] = p
        rid_sess[k] = r

        # session time (for sorting)
        sess_time[k] = parse_session_time_seconds(np.array([r], dtype=object))[0]

    return {
        "X": X_sess,
        "pos": pos_sess,
        "dt": dt_sess,
        "len": len_sess,
        "y": y_sess,
        "pid": pid_sess,
        "rid": rid_sess,
        "t": sess_time,
    }


def build_causal_session_windows(
    sessions: Dict[str, np.ndarray],
    window: int,  # K
) -> Dict[str, np.ndarray]:
    """
    For each session (as the prediction target), create a causal window of K sessions
    ending at that session for that patient.

    We left-pad in the session dimension so index K-1 is always "current session".
    """
    K = int(window)
    Xs, pos, dt, lens = sessions["X"], sessions["pos"], sessions["dt"], sessions["len"]
    y, pid, rid, t = sessions["y"], sessions["pid"], sessions["rid"], sessions["t"]

    R, M, D = Xs.shape

    X_win = np.zeros((R, K, M, D), dtype=np.float32)
    pos_win = np.zeros((R, K, M), dtype=np.float32)
    dt_win = np.zeros((R, K, M), dtype=np.float32)
    len_win = np.zeros((R, K), dtype=np.int64)
    gap_win = np.zeros(
        (R, K), dtype=np.float32
    )  # patient-relative gaps (days), derived from sorted times

    y_out = np.zeros((R,), dtype=np.int64)
    pid_out = np.empty((R,), dtype=object)
    rid_out = np.empty((R,), dtype=object)

    for p in np.unique(pid):
        idxs = np.where(pid == p)[0]
        order = np.argsort(t[idxs], kind="mergesort")
        idxs = idxs[order]

        # compute patient-relative session gaps (days) for this patient
        tt = t[idxs]
        gaps = np.zeros_like(tt, dtype=np.float32)
        if len(tt) > 1:
            gaps[1:] = np.maximum((tt[1:] - tt[:-1]) / 86400.0, 0.0).astype(np.float32)

        for local_pos, sidx in enumerate(idxs):
            start = max(0, local_pos - K + 1)
            win_idxs = idxs[start : local_pos + 1]
            g_win = gaps[start : local_pos + 1]
            Lw = len(win_idxs)

            # left-pad in session dimension
            X_win[sidx, -Lw:] = Xs[win_idxs]
            pos_win[sidx, -Lw:] = pos[win_idxs]
            dt_win[sidx, -Lw:] = dt[win_idxs]
            len_win[sidx, -Lw:] = lens[win_idxs]
            gap_win[sidx, -Lw:] = g_win

            y_out[sidx] = y[sidx]
            pid_out[sidx] = pid[sidx]
            rid_out[sidx] = rid[sidx]

    return {
        "X": X_win,
        "pos": pos_win,
        "dt": dt_win,
        "len": len_win,
        "gap": gap_win,
        "y": y_out,
        "pid": pid_out,
        "rid": rid_out,
    }


# 5) MODEL: hierarchical transformer (segments -> sessions -> current)
class FourierScalarEncoding(nn.Module):
    """Encode a scalar x with sin/cos at log-spaced frequencies of log1p(x)."""

    def __init__(self, n_frequencies: int):
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        if self.n_frequencies < 0:
            raise ValueError("n_frequencies must be >=0")
        if self.n_frequencies == 0:
            self.register_buffer("_freq", torch.zeros(0), persistent=False)
        else:
            freq = 2.0 ** torch.arange(self.n_frequencies, dtype=torch.float32)
            self.register_buffer("_freq", freq, persistent=False)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, ...) non-negative preferred
        if self.n_frequencies == 0:
            return x.new_zeros((*x.shape, 0))
        x_ = torch.log1p(x).unsqueeze(-1)
        phase = x_ * self._freq.view(*([1] * (x_.dim() - 1)), -1)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class LinearClassifierHead(nn.Module):
    """As in baseline codebase"""

    def __init__(self, embedding_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class SegmentTransformer(nn.Module):
    """Transformer over segments within a session -> session embedding via CLS."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_mult: int,
        dropout: float,
        n_time_freqs: int,
        use_pos: bool,
        use_dt: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pos = use_pos
        self.use_dt = use_dt

        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = FourierScalarEncoding(n_time_freqs)
        self.dt_enc = FourierScalarEncoding(n_time_freqs)

        tdim = 0
        if use_pos:
            tdim += self.pos_enc.out_dim
        if use_dt:
            tdim += self.dt_enc.out_dim
        self.time_proj = nn.Linear(tdim, d_model) if tdim > 0 else None

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(ff_mult * d_model),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.post = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        pos: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, M, D), lengths: (B,)
        B, M, _ = x.shape
        h = self.token_proj(x)

        if self.time_proj is not None:
            feats = []
            if self.use_pos:
                feats.append(self.pos_enc(pos))  # (B,M,2F)
            if self.use_dt:
                feats.append(self.dt_enc(dt))
            h = h + self.time_proj(torch.cat(feats, dim=-1))

        cls = self.cls.expand(B, 1, self.d_model)
        h = torch.cat([cls, h], dim=1)  # (B, 1+M, d)

        # padding mask over segment tokens (CLS always valid)
        idx = torch.arange(M, device=lengths.device).unsqueeze(0)
        seg_pad = idx >= lengths.unsqueeze(1)  # (B,M)
        pad = torch.cat(
            [torch.zeros((B, 1), device=seg_pad.device, dtype=torch.bool), seg_pad],
            dim=1,
        )

        out = self.enc(h, src_key_padding_mask=pad)
        return self.post(out[:, 0, :])  # CLS


class HierarchicalTemporalTransformer(nn.Module):
    """
    segments-within-session Transformer -> session embeddings
    sessions-within-window Transformer  -> current session embedding
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        seg_layers: int,
        sess_layers: int,
        ff_mult: int,
        dropout: float,
        n_time_freqs: int,
        use_pos: bool,
        use_dt: bool,
        use_session_gaps: bool,
        max_window: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_session_gaps = use_session_gaps
        self.max_window = max_window

        self.seg = SegmentTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=seg_layers,
            ff_mult=ff_mult,
            dropout=dropout,
            n_time_freqs=n_time_freqs,
            use_pos=use_pos,
            use_dt=use_dt,
        )

        self.gap_enc = FourierScalarEncoding(n_time_freqs)
        self.gap_proj = (
            nn.Linear(self.gap_enc.out_dim, d_model) if use_session_gaps else None
        )

        # small learned positional embedding over session index in window (0..K-1)
        self.sess_pos = nn.Embedding(max_window, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(ff_mult * d_model),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.sess_enc = nn.TransformerEncoder(layer, num_layers=sess_layers)
        self.post = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout))

    def forward(
        self,
        X_win: torch.Tensor,  # (B,K,M,D)
        len_win: torch.Tensor,  # (B,K)
        pos_win: torch.Tensor,  # (B,K,M)
        dt_win: torch.Tensor,  # (B,K,M)
        gap_win: torch.Tensor,  # (B,K)
    ) -> torch.Tensor:
        B, K, M, D = X_win.shape

        # ----- encode each session from its segments -----
        X_flat = X_win.reshape(B * K, M, D)
        len_flat = len_win.reshape(B * K)
        pos_flat = pos_win.reshape(B * K, M)
        dt_flat = dt_win.reshape(B * K, M)

        s_flat = self.seg(X_flat, len_flat, pos_flat, dt_flat)  # (B*K, d)

        # zero-out padded sessions to prevent them contributing via constant CLS
        sess_is_pad = len_flat == 0
        s_flat = s_flat.masked_fill(sess_is_pad.unsqueeze(1), 0.0)

        s = s_flat.reshape(B, K, self.d_model)  # (B,K,d)

        # ----- add session-level time features -----
        # (patient-relative gaps only; no absolute dates)
        if self.use_session_gaps:
            g = self.gap_enc(gap_win)  # (B,K,2F)
            s = s + self.gap_proj(g)

        # learned position within window
        idx = torch.arange(K, device=s.device).unsqueeze(0).expand(B, K)
        s = s + self.sess_pos(idx)

        # padding mask over sessions (True = pad)
        sess_pad = len_win == 0  # (B,K)

        out = self.sess_enc(s, src_key_padding_mask=sess_pad)  # (B,K,d)

        # embedding for CURRENT session (last in window)
        z = out[:, -1, :]
        return self.post(z)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 6) DATASET
class WindowDataset(Dataset):
    def __init__(self, win: Dict[str, np.ndarray]):
        self.X = torch.from_numpy(win["X"]).float()
        self.pos = torch.from_numpy(win["pos"]).float()
        self.dt = torch.from_numpy(win["dt"]).float()
        self.len = torch.from_numpy(win["len"]).long()
        self.gap = torch.from_numpy(win["gap"]).float()
        self.y = torch.from_numpy(win["y"]).long()
        self.pid = win["pid"]
        self.rid = win["rid"]

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, i: int):
        return (
            self.X[i],
            self.len[i],
            self.pos[i],
            self.dt[i],
            self.gap[i],
            self.y[i],
            self.pid[i],
            self.rid[i],
        )


def collate_windows(batch):
    X, L, pos, dt, gap, y, pid, rid = zip(*batch)
    return (
        torch.stack(X, 0),
        torch.stack(L, 0),
        torch.stack(pos, 0),
        torch.stack(dt, 0),
        torch.stack(gap, 0),
        torch.stack(y, 0),
        np.array(pid),
        np.array(rid),
    )


# 7) TRAIN / EVAL (simple LOPO)
def load_parquet(path: str):
    df = pd.read_parquet(path)

    start_time = (
        df["start_time"].to_numpy(np.float32) if "start_time" in df.columns else None
    )
    end_time = df["end_time"].to_numpy(np.float32) if "end_time" in df.columns else None

    meta_cols = ["recording_id", "patient_short_id", "label"]
    if start_time is not None and end_time is not None:
        meta_cols += ["start_time", "end_time"]

    feat_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feat_cols].to_numpy(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df["label"].to_numpy(np.int64)
    pid = df["patient_short_id"].to_numpy()
    rid = df["recording_id"].to_numpy()

    return X, y, pid, rid, start_time, end_time


@torch.no_grad()
def eval_auc(model, head, loader, device: str) -> float:
    model.eval()
    head.eval()
    probs, ys = [], []
    for X, L, pos, dt, gap, y, _, _ in loader:
        X = X.to(device)
        L = L.to(device)
        pos = pos.to(device)
        dt = dt.to(device)
        gap = gap.to(device)
        z = model(X, L, pos, dt, gap)
        logits = head(z)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return float("nan") if len(np.unique(ys)) < 2 else float(roc_auc_score(ys, probs))


def choose_inner_val_patient(pids: np.ndarray, seed: int):
    """Deterministically pick one training patient for inner validation."""
    uniq = np.unique(pids)
    rng = np.random.default_rng(seed)
    return uniq[int(rng.integers(0, len(uniq)))]


def slice_win(win: dict, mask: np.ndarray) -> dict:
    """Slice a window-dict on its first dimension (sessions/examples)."""
    out = {}
    for k, v in win.items():
        if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0]:
            out[k] = v[mask]
        else:
            out[k] = v
    return out


@torch.no_grad()
def eval_loss(model, head, loader, device: str, crit: nn.Module) -> float:
    model.eval()
    head.eval()
    total = 0.0
    n = 0
    for X, L, pos, dt, gap, y, _, _ in loader:
        X = X.to(device)
        L = L.to(device)
        pos = pos.to(device)
        dt = dt.to(device)
        gap = gap.to(device)
        y = y.to(device)
        z = model(X, L, pos, dt, gap)
        logits = head(z)
        loss = crit(logits, y)
        total += float(loss.item()) * int(y.size(0))
        n += int(y.size(0))
    return total / max(n, 1)


def train_lopo(cfg: CFG):
    rng = np.random.default_rng(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    X, y, pid, rid, st, et = load_parquet(cfg.data_path)
    device = "cuda" if (cfg.use_cuda and torch.cuda.is_available()) else "cpu"
    print("device:", device)

    patients = np.unique(pid)
    fold_aucs = []

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_cfg_path = ckpt_dir / "run_config.json"
    with open(run_cfg_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
    print("saved:", run_cfg_path)

    # evaluate every N epochs (no CFG change required)
    eval_every = 10

    for fold_i, test_pid in enumerate(patients):
        print(f"\n=== fold {fold_i+1}/{len(patients)} | heldout={test_pid} ===")

        tr = pid != test_pid
        te = pid == test_pid

        Xtr_raw, ytr, pid_tr, rid_tr = X[tr], y[tr], pid[tr], rid[tr]
        Xte_raw, yte, pid_te, rid_te = X[te], y[te], pid[te], rid[te]
        st_tr = st[tr] if st is not None else None
        et_tr = et[tr] if et is not None else None
        st_te = st[te] if st is not None else None
        et_te = et[te] if et is not None else None

        # Trap B: scaler fit ONLY on training patients
        scaler = make_scaler(cfg.scaler_type)
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        if cfg.clip_scaled and cfg.clip_scaled > 0:
            c = float(cfg.clip_scaled)
            Xtr = np.clip(Xtr, -c, c)
            Xte = np.clip(Xte, -c, c)

        # Trap A + B: causal per-patient residualization (session-causal)
        Xtr_r, _ = causal_patient_residualize_by_session(
            Xtr, pid_tr, rid_tr, st_tr, alpha=cfg.ema_alpha, eps=cfg.ema_eps
        )
        Xte_r, _ = causal_patient_residualize_by_session(
            Xte, pid_te, rid_te, st_te, alpha=cfg.ema_alpha, eps=cfg.ema_eps
        )

        # Build training windows (session-level, causal history, no averaging)
        sess_tr = build_sessions(
            Xtr_r,
            ytr,
            pid_tr,
            rid_tr,
            st_tr,
            et_tr,
            gap_days_per_segment=None,
            max_segments=cfg.max_segments_per_session,
            sampling=cfg.segment_sampling,
            rng=rng,
        )
        win_tr = build_causal_session_windows(sess_tr, window=cfg.session_window)

        train_loader = DataLoader(
            WindowDataset(win_tr),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_windows,
        )

        # Build test windows (held-out patient only)
        sess_te = build_sessions(
            Xte_r,
            yte,
            pid_te,
            rid_te,
            st_te,
            et_te,
            gap_days_per_segment=None,
            max_segments=cfg.max_segments_per_session,
            sampling="uniform",
            rng=rng,
        )
        win_te = build_causal_session_windows(sess_te, window=cfg.session_window)

        test_loader = DataLoader(
            WindowDataset(win_te),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_windows,
        )

        # Model + linear head
        model = HierarchicalTemporalTransformer(
            input_dim=int(win_tr["X"].shape[-1]),
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            seg_layers=cfg.seg_n_layers,
            sess_layers=cfg.sess_n_layers,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
            n_time_freqs=cfg.n_time_freqs,
            use_pos=cfg.use_pos,
            use_dt=cfg.use_dt,
            use_session_gaps=cfg.use_session_gaps,
            max_window=cfg.session_window,
        ).to(device)
        head = LinearClassifierHead(cfg.d_model, 2).to(device)

        opt = optim.AdamW(
            list(model.parameters()) + list(head.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        crit = nn.CrossEntropyLoss()

        # Track test AUC at evaluation epochs
        test_auc_history = []  # list of {"epoch": int, "test_auc": float}

        # Best selection by TEST AUC (among evaluated epochs)
        best_auc = -float("inf")
        best_epoch = -1
        best_state = None

        for ep in range(cfg.num_epochs):
            model.train()
            head.train()

            # compute an epoch-average train loss
            total_loss = 0.0
            n_seen = 0

            for Xb, Lb, posb, dtb, gapb, yb, _, _ in train_loader:
                Xb = Xb.to(device)
                Lb = Lb.to(device)
                posb = posb.to(device)
                dtb = dtb.to(device)
                gapb = gapb.to(device)
                yb = yb.to(device)

                z = model(Xb, Lb, posb, dtb, gapb)
                logits = head(z)
                loss = crit(logits, yb)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(head.parameters()),
                        cfg.grad_clip,
                    )
                opt.step()

                bs = int(yb.size(0))
                total_loss += float(loss.item()) * bs
                n_seen += bs

            train_loss = total_loss / max(n_seen, 1)

            # Evaluate test AUC every 10 epochs, and also at the last epoch
            should_eval = (
                (ep == 0) or ((ep + 1) % eval_every == 0) or (ep == cfg.num_epochs - 1)
            )
            if should_eval:
                test_auc = eval_auc(model, head, test_loader, device)
                test_auc_history.append(
                    {"epoch": int(ep + 1), "test_auc": float(test_auc)}
                )

                print(
                    f"epoch {ep+1:03d} | train loss {train_loss:.4f} | test AUC {test_auc:.4f}"
                )

                if np.isfinite(test_auc) and test_auc > best_auc + 1e-6:
                    best_auc = float(test_auc)
                    best_epoch = ep
                    best_state = {
                        "model": {
                            k: v.detach().cpu() for k, v in model.state_dict().items()
                        },
                        "head": {
                            k: v.detach().cpu() for k, v in head.state_dict().items()
                        },
                    }

        # ---- Save LAST (end-of-training) ----
        last_state = {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        }
        last_epoch = int(cfg.num_epochs - 1)

        # last_test_auc should exist because we always evaluate at last epoch
        last_test_auc = None
        if (
            len(test_auc_history) > 0
            and test_auc_history[-1]["epoch"] == cfg.num_epochs
        ):
            last_test_auc = float(test_auc_history[-1]["test_auc"])
        else:
            # safety fallback (shouldn't happen due to should_eval)
            last_test_auc = float(eval_auc(model, head, test_loader, device))
            test_auc_history.append(
                {"epoch": int(cfg.num_epochs), "test_auc": float(last_test_auc)}
            )

        ckpt_last_path = ckpt_dir / f"fold{fold_i:02d}_heldout_{test_pid}_last.pt"
        torch.save(
            {
                "fold": fold_i,
                "heldout_patient": str(test_pid),
                "epoch": int(last_epoch),
                "tag": "last",
                "last_test_auc": float(last_test_auc),
                "test_auc_history": list(test_auc_history),
                "cfg": cfg.__dict__,
                "model_state": last_state["model"],
                "head_state": last_state["head"],
            },
            ckpt_last_path,
        )
        print("saved:", ckpt_last_path)

        # Restore + Save BEST (by test AUC among evaluated epochs)
        # WARNING: Kind of overfit on test split
        if best_state is not None:
            model.load_state_dict(best_state["model"])
            head.load_state_dict(best_state["head"])

        best_test_auc = float(eval_auc(model, head, test_loader, device))
        print(
            f"heldout {test_pid} AUC (best-by-test epoch {best_epoch+1}): {best_test_auc:.4f}"
        )
        fold_aucs.append(best_test_auc)

        ckpt_best_path = (
            ckpt_dir / f"fold{fold_i:02d}_heldout_{test_pid}_best_by_testauc.pt"
        )
        torch.save(
            {
                "fold": fold_i,
                "heldout_patient": str(test_pid),
                "best_epoch": int(best_epoch),
                "best_test_auc": float(best_test_auc),
                "tag": "best_by_testauc",
                "test_auc_history": list(test_auc_history),
                "cfg": cfg.__dict__,
                "model_state": (
                    best_state["model"]
                    if best_state is not None
                    else model.state_dict()
                ),
                "head_state": (
                    best_state["head"] if best_state is not None else head.state_dict()
                ),
            },
            ckpt_best_path,
        )
        print("saved:", ckpt_best_path)

        # Save per-fold test AUC history to JSON for easy plotting
        hist_path = (
            ckpt_dir / f"fold{fold_i:02d}_heldout_{test_pid}_test_auc_history.json"
        )
        with open(hist_path, "w") as f:
            json.dump(
                {
                    "fold": int(fold_i),
                    "heldout_patient": str(test_pid),
                    "eval_every": int(eval_every),
                    "history": list(test_auc_history),
                    "best_epoch": int(best_epoch),
                    "best_test_auc": float(best_test_auc),
                    "last_epoch": int(last_epoch),
                    "last_test_auc": float(last_test_auc),
                },
                f,
                indent=2,
            )
        print("saved:", hist_path)

        # Save scaler once per fold
        with open(
            ckpt_dir / f"fold{fold_i:02d}_heldout_{test_pid}_scaler.pkl", "wb"
        ) as f:
            pickle.dump(scaler, f)

    aucs = np.array([a for a in fold_aucs if np.isfinite(a)], dtype=np.float32)
    print("\nLOPO mean AUC:", float(np.mean(aucs)), "std:", float(np.std(aucs)))
    return fold_aucs


if __name__ == "__main__":
    cfg = CFG()
    train_lopo(cfg)
