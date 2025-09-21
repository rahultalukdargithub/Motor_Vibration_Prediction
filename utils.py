import numpy as np
from PyEMD import EMD
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm
import glob
from scipy.io import loadmat
import pandas as pd
import os

FS = 1000.0
WIN_SAMPLES = 5000

def emd_denoise(signal, max_imfs=10):
    emd = EMD()
    emd.FIXE = 0
    emd.MAX_ITERATION = 100
    emd.spline_kind = "cubic"
    pad = 50
    s_pad = np.r_[signal[pad:0:-1], signal, signal[-2:-pad-2:-1]]
    imfs = emd.emd(s_pad)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    imfs = imfs[:, pad:pad+len(signal)]
    if imfs.shape[0] > max_imfs:
        imfs = imfs[:max_imfs]
    elif imfs.shape[0] < max_imfs:
        imfs = np.vstack([imfs, np.zeros((max_imfs - imfs.shape[0], len(signal)))])
    recon_all = np.sum(imfs, axis=0)
    residue = signal - recon_all
    recon = np.sum(imfs[1:], axis=0) + residue
    return recon

def features_time(sig):
    M = np.mean(sig)
    SD = np.std(sig, ddof=1)
    SKw = skew(sig, bias=False)
    KRt = kurtosis(sig, fisher=False, bias=False)
    PP = np.max(sig) - np.min(sig)
    RMS = np.sqrt(np.mean(sig**2))
    E = np.sum(sig**2)
    return [M, SD, SKw, KRt, PP, RMS, E]

def features_freq(sig, fs=FS):
    sig_zm = sig - np.mean(sig)
    X = np.abs(rfft(sig_zm))
    freqs = rfftfreq(sig_zm.size, d=1.0/fs)
    psd = X**2
    power_sum = np.sum(psd) + 1e-12
    FM = np.sum(freqs * psd) / power_sum
    var_f = np.sum(((freqs - FM)**2) * psd) / power_sum
    FSD = np.sqrt(max(var_f, 0.0))
    p = psd / power_sum
    centered = freqs - FM
    m3 = np.sum((centered**3) * p)
    m4 = np.sum((centered**4) * p)
    FSK = m3 / (FSD**3 + 1e-12)
    FKR = m4 / (FSD**4 + 1e-12)
    BPWR_N = power_sum / sig_zm.size
    cdf = np.cumsum(p)
    idx_med = np.searchsorted(cdf, 0.5)
    FMED = freqs[min(idx_med, len(freqs)-1)]
    return [FM, FSD, FSK, FKR, BPWR_N, FMED]

def extract_F4(sig):
    return np.array(features_time(sig) + features_freq(sig), dtype=float)

def combine_axes_normalize(x, y, z):
    s = np.sqrt(x**2 + y**2 + z**2)
    max_abs = np.max(np.abs(s))
    if max_abs > 0:
        s = s / max_abs
    return s



def tune_svm_rbf(X_feats_h, y_h, n_splits=5, random_state=42):
    """
    Trains a OneClassSVM with RBF kernel using stratified k-fold CV on healthy-only samples.
    Returns the final fitted pipeline on all healthy samples using best params from CV.
    
    Parameters:
    - X_feats_h: feature matrix for healthy samples only
    - y_h: labels for healthy samples (all zeros)
    - n_splits: number of CV folds
    """
    print(f"Starting One-Class SVM RBF tuning on {X_feats_h.shape[0]} healthy samples...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    nu_list = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
    gamma_list = ["scale", 0.01, 0.05, 0.1, 0.2, 0.5]

    best_score = -1
    best_params = None

    total_combinations = len(nu_list) * len(gamma_list)
    comb_counter = 1

    # Manual grid search
    for nu in nu_list:
        for gamma in gamma_list:
            print(f"Testing combination {comb_counter}/{total_combinations}: nu={nu}, gamma={gamma}")
            comb_counter += 1
            fold_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_feats_h, y_h), start=1):
                X_train_fold = X_feats_h[train_idx]
                X_val_fold = X_feats_h[val_idx]  # evaluation on healthy fold

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ocsvm", OneClassSVM(kernel="rbf", gamma=gamma, nu=nu))
                ])
                pipe.fit(X_train_fold)
                preds = pipe.predict(X_val_fold)
                pred_labels = np.where(preds == 1, 0, 1)  # 0=healthy, 1=faulty
                acc = accuracy_score(y_h[val_idx], pred_labels)
                fold_scores.append(acc)
                print(f"  Fold {fold_idx} accuracy: {acc:.4f}")

            mean_score = np.mean(fold_scores)
            print(f"  Mean CV accuracy for nu={nu}, gamma={gamma}: {mean_score:.4f}\n")

            if mean_score > best_score:
                best_score = mean_score
                best_params = {"nu": nu, "gamma": gamma}

    print("Best params (OCSVM RBF, CV):", best_params, "Best CV acc:", best_score)

    # Refit final model on all healthy samples using best params
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf", **best_params))
    ])
    final_pipe.fit(X_feats_h)
    print("Final One-Class SVM trained on all healthy samples.\n")
    
    return final_pipe



COMMON_AXIS_KEYS = [
    ('x','y','z'),
    ('X','Y','Z'),
    ('ax','ay','az'),
    ('AX','AY','AZ'),
    ('ch1','ch2','ch3'),
    ('CH1','CH2','CH3'),
    ('vx','vy','vz'),
    ('a_x','a_y','a_z'),
]

def _find_axes_in_mat(mat):
    keys = set(mat.keys())
    if 'H' in keys:
        H = np.squeeze(np.array(mat['H'])).astype(float)
        if isinstance(H, np.ndarray) and H.ndim == 2 and H.shape[1] == 3:
            return H[:,0], H[:,1], H[:,2]
    for kx,ky,kz in COMMON_AXIS_KEYS:
        if kx in keys and ky in keys and kz in keys:
            x = np.squeeze(mat[kx]).astype(float)
            y = np.squeeze(mat[ky]).astype(float)
            z = np.squeeze(mat[kz]).astype(float)
            return x, y, z
    arrays = [np.squeeze(mat[k]) for k in keys if not k.startswith('__')]
    vecs = [a for a in arrays if isinstance(a, np.ndarray) and a.ndim == 1]
    if len(vecs) >= 3 and len(vecs[0]) == len(vecs[1]) == len(vecs[2]):
        return vecs[0].astype(float), vecs[1].astype(float), vecs[2].astype(float)
    raise ValueError("Tri-axial vectors not found")

def _segment_or_trim(sig, target_len=WIN_SAMPLES):
    n = len(sig)
    if n == target_len:
        return [sig]
    if n < target_len:
        out = np.zeros(target_len, dtype=float)
        out[:n] = sig
        return [out]
    segments = []
    start = 0
    while start + target_len <= n:
        segments.append(sig[start:start+target_len])
        start += target_len
    return segments

def load_folder_mat(folder, label):
    Xx_list, Xy_list, Xz_list, y_list = [], [], [], []
    files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    for f in tqdm(files, desc=f"Loading {folder}"):
        mat = loadmat(f)
        x, y, z = _find_axes_in_mat(mat)
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
        x_segs = _segment_or_trim(x, WIN_SAMPLES)
        y_segs = _segment_or_trim(y, WIN_SAMPLES)
        z_segs = _segment_or_trim(z, WIN_SAMPLES)
        n_segs = min(len(x_segs), len(y_segs), len(z_segs))
        for i in range(n_segs):
            Xx_list.append(x_segs[i])
            Xy_list.append(y_segs[i])
            Xz_list.append(z_segs[i])
            y_list.append(label)
    return np.array(Xx_list), np.array(Xy_list), np.array(Xz_list), np.array(y_list, dtype=int)


def build_dataset_features(Xx, Xy, Xz):
    N = Xx.shape[0]
    feats = np.zeros((N, 13), dtype=float)
    for i in tqdm(range(N), desc="EMD+F4 features"):
        s = combine_axes_normalize(Xx[i], Xy[i], Xz[i])
        s_d = emd_denoise(s, max_imfs=10)
        feats[i] = extract_F4(s_d)
    cols = ["M","SD","SK","KR","PP","RMS","E","FM","FSD","FSK","FKR","BPWR_N","FMED"]
    return pd.DataFrame(feats, columns=cols)