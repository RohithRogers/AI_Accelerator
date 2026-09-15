#!/usr/bin/env python3
"""
flow/har_dataset.py — HAR Dataset Utilities for TinyML Accelerator Testing
===========================================================================

PURPOSE:
  Downloads and preprocesses the UCI Human Activity Recognition (HAR) dataset,
  extracting a compact set of 16 statistical features per 128-sample window.
  Provides a synthetic-data fallback (no internet required) for CI environments.

FEATURE VECTOR (16 floats):
  For each of 4 sensor channels [accel_x, accel_y, accel_z, gyro_magnitude]:
    - mean           (DC component)
    - std            (spread / energy proxy)
    - energy         (sum of squares, normalised by window length)
    - peak-to-peak   (range)

LABELS:
  0=WALKING, 1=WALKING_UPSTAIRS, 2=WALKING_DOWNSTAIRS,
  3=SITTING,  4=STANDING,          5=LAYING

UCI HAR reference:
  D. Anguita et al., 2013 - https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
"""

import os
import sys
import zipfile
import urllib.request
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CACHE_DIR  = os.path.join(ROOT_DIR, "temp", "har_cache")

HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)
HAR_ZIP  = os.path.join(CACHE_DIR, "UCI_HAR_Dataset.zip")
HAR_ROOT = os.path.join(CACHE_DIR, "UCI HAR Dataset")

N_FEATURES  = 16      # output feature vector length
N_CLASSES   = 6       # activity classes
WINDOW_LEN  = 128     # raw samples per window in UCI HAR
N_CHANNELS  = 9       # raw signal channels in UCI HAR

CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING",  "STANDING",          "LAYING"]

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, int(downloaded * 100 / total_size))
        bar = "=" * (pct // 5) + " " * (20 - pct // 5)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ({downloaded // 1024} KB)")
        sys.stdout.flush()
    if downloaded >= total_size:
        print()


def download_uci_har(verbose=True):
    """Download and unzip UCI HAR into CACHE_DIR if not already present."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.isdir(HAR_ROOT):
        if verbose:
            print(f"[HAR] Dataset already cached at: {HAR_ROOT}")
        return HAR_ROOT

    if not os.path.isfile(HAR_ZIP):
        if verbose:
            print(f"[HAR] Downloading UCI HAR dataset from:\n  {HAR_URL}")
        try:
            urllib.request.urlretrieve(HAR_URL, HAR_ZIP, reporthook=_reporthook)
        except Exception as exc:
            raise RuntimeError(
                f"Download failed: {exc}\n"
                "Check internet connectivity or use generate_synthetic_har() instead."
            ) from exc
        if verbose:
            print(f"[HAR] Saved zip to: {HAR_ZIP}")

    if verbose:
        print(f"[HAR] Extracting to: {CACHE_DIR}")
    with zipfile.ZipFile(HAR_ZIP, "r") as zf:
        zf.extractall(CACHE_DIR)

    if verbose:
        print(f"[HAR] Extraction complete.")
    return HAR_ROOT


# ---------------------------------------------------------------------------
# Raw signal loader
# ---------------------------------------------------------------------------

def _load_raw_signals(split_dir):
    """
    Load all 9 raw Inertial Signals for a given split directory.
    Returns array of shape (N_windows, WINDOW_LEN, N_CHANNELS).
    """
    inertial_dir = os.path.join(split_dir, "Inertial Signals")
    signal_files = sorted([
        f for f in os.listdir(inertial_dir)
        if f.endswith(".txt")
    ])

    if len(signal_files) == 0:
        raise FileNotFoundError(f"No signal files found in: {inertial_dir}")

    channels = []
    for fname in signal_files:
        fpath = os.path.join(inertial_dir, fname)
        data = np.loadtxt(fpath)   # shape: (N_windows, WINDOW_LEN)
        channels.append(data)

    # Stack to (N_windows, WINDOW_LEN, N_channels)
    return np.stack(channels, axis=-1).astype(np.float32)


def _load_labels(split_dir, split_name):
    """Load integer activity labels (1-indexed). Returns 0-indexed array."""
    label_file = os.path.join(split_dir, f"y_{split_name}.txt")
    labels = np.loadtxt(label_file, dtype=np.int32)
    return labels - 1   # convert 1-6 to 0-5


# ---------------------------------------------------------------------------
# Feature extraction: 128-sample window to 16-element float vector
# ---------------------------------------------------------------------------

def extract_features(window):
    """
    Extract 16 statistical features from a (WINDOW_LEN, N_channels) signal window.

    We use 4 derived channels:
      ch0 = body_acc_x (channel index 3)
      ch1 = body_acc_y (channel index 4)
      ch2 = body_acc_z (channel index 5)
      ch3 = gyro_magnitude = sqrt(gx^2 + gy^2 + gz^2)  (channels 6,7,8)

    Per channel: [mean, std, energy, peak-to-peak] gives 4 x 4 = 16 features.
    """
    ax = window[:, 3]    # body_acc_x
    ay = window[:, 4]    # body_acc_y
    az = window[:, 5]    # body_acc_z
    gx = window[:, 6]
    gy = window[:, 7]
    gz = window[:, 8]
    gm = np.sqrt(gx**2 + gy**2 + gz**2)   # gyro magnitude

    feats = []
    for ch in [ax, ay, az, gm]:
        feats.append(float(np.mean(ch)))
        feats.append(float(np.std(ch)))
        feats.append(float(np.mean(ch**2)))          # energy (normalised)
        feats.append(float(np.max(ch) - np.min(ch))) # peak-to-peak
    return np.array(feats, dtype=np.float32)


def _build_feature_matrix(raw_signals):
    """Apply extract_features to every window. Returns (N, N_FEATURES)."""
    return np.stack([extract_features(raw_signals[i])
                     for i in range(raw_signals.shape[0])], axis=0)


# ---------------------------------------------------------------------------
# Public API: load_uci_har()
# ---------------------------------------------------------------------------

def load_uci_har(verbose=True):
    """
    Download (if necessary) and load UCI HAR.
    Returns:
        X_train, y_train : float32 (N_train, N_FEATURES), int32 (N_train,)
        X_test,  y_test  : float32 (N_test,  N_FEATURES), int32 (N_test,)
    """
    har_root = download_uci_har(verbose=verbose)

    train_dir = os.path.join(har_root, "train")
    test_dir  = os.path.join(har_root, "test")

    if verbose:
        print("[HAR] Loading raw train signals...")
    raw_train = _load_raw_signals(train_dir)
    y_train   = _load_labels(train_dir, "train")

    if verbose:
        print("[HAR] Loading raw test signals...")
    raw_test  = _load_raw_signals(test_dir)
    y_test    = _load_labels(test_dir, "test")

    if verbose:
        print("[HAR] Extracting features...")
    X_train = _build_feature_matrix(raw_train)
    X_test  = _build_feature_matrix(raw_test)

    if verbose:
        print(f"[HAR] Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"[HAR] Classes: {N_CLASSES} -- {CLASS_NAMES}")

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Synthetic fallback: Gaussian blobs with realistic inter-class separation
# ---------------------------------------------------------------------------

_SYNTHETIC_MEANS = np.array([
    # mean, std, energy, p2p for [ax, ay, az, gm]
    [ 0.05, 0.20, 0.10, 0.80, -0.02, 0.18, 0.09, 0.75,  0.00, 0.22, 0.10, 0.90,  2.5, 1.5,  8.0, 6.0],  # WALKING
    [ 0.10, 0.28, 0.18, 1.20, -0.05, 0.25, 0.13, 1.00,  0.03, 0.30, 0.16, 1.25,  3.5, 2.0, 12.0, 8.0],  # WALK_UP
    [ 0.08, 0.25, 0.14, 1.00, -0.04, 0.22, 0.11, 0.85,  0.01, 0.27, 0.13, 1.10,  3.0, 1.8, 10.0, 7.5],  # WALK_DOWN
    [-0.01, 0.03, 0.01, 0.10,  0.00, 0.02, 0.01, 0.08, -0.01, 0.03, 0.01, 0.12,  0.2, 0.1,  0.1, 0.4],  # SITTING
    [ 0.00, 0.02, 0.00, 0.08,  0.00, 0.02, 0.00, 0.06,  0.00, 0.02, 0.00, 0.09,  0.1, 0.05, 0.05, 0.2], # STANDING
    [ 0.00, 0.01, 0.00, 0.04,  0.00, 0.01, 0.00, 0.03,  0.00, 0.01, 0.00, 0.05, 0.05, 0.02, 0.02, 0.1], # LAYING
], dtype=np.float32)

_SYNTHETIC_STD = 0.03   # isotropic noise; separable but non-trivial classes


def generate_synthetic_har(n_samples_per_class=100, seed=42):
    """
    Generate synthetic HAR feature vectors as labelled Gaussian blobs.
    Returns X (N, 16), y (N,) matching the same interface as load_uci_har().
    Total N = n_samples_per_class * N_CLASSES.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for cls_idx, mean in enumerate(_SYNTHETIC_MEANS):
        noise = rng.normal(0, _SYNTHETIC_STD,
                           size=(n_samples_per_class, N_FEATURES)).astype(np.float32)
        X_list.append(mean + noise)
        y_list.append(np.full(n_samples_per_class, cls_idx, dtype=np.int32))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Z-score normalisation helpers
# ---------------------------------------------------------------------------

def fit_normalizer(X_train):
    """Compute per-feature mean and std from training set."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8   # avoid division by zero
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Synthetic HAR smoke test ===")
    X, y = generate_synthetic_har(n_samples_per_class=20, seed=0)
    print(f"Shape: {X.shape}, Labels: {np.unique(y)}")
    print(f"Feature[0]: {X[0]}")
    print(f"Label  [0]: {CLASS_NAMES[y[0]]}")

    mean, std = fit_normalizer(X)
    X_norm = normalize(X, mean, std)
    print(f"Normalized mean approx 0: {X_norm.mean(axis=0).round(3)}")
    print(f"Normalized std  approx 1: {X_norm.std(axis=0).round(3)}")
    print("=== Synthetic test PASSED ===")
