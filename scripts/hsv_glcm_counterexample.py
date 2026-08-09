# scripts/find_hsv_glcm_counterexample.py

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

# ============================================================
# EDIT BAGIAN INI SAJA
# ============================================================

CONFIG_PATH = "config/config_raw.yaml"

# Untuk visualisasi counterexample, disarankan RAW dulu.
# Alasannya: full segmented image bisa didominasi piksel 0,0 dari masking.

FORCE_RECOMPUTE_FEATURES = False

TOP_N_CANDIDATES = 20

# Filter kandidat
HSV_QUANTILE = 0.10       # HSV harus termasuk 10% paling mirip
TARGET_QUANTILE = 0.75    # beda target harus termasuk 25% paling besar

# Ambil kandidat ke berapa dari hasil sorting
SELECT_CANDIDATE_ORDER = 702

# Output
OUTPUT_DIR = "results/figures/counterexample_hsv_glcm"
SAVE_CANDIDATES_CSV = True
SAVE_VISUALIZATION = True

# Histogram visualisasi HSV sederhana
HIST_BINS_H = 36
HIST_BINS_S = 32
HIST_BINS_V = 32

# ============================================================


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config as cfg_module
from data.loader import load_and_prepare_data
from data.preprocessor import ImagePreprocessor
from features.extractor import FeatureExtractor


def load_or_extract_features(config, df_valid):
    feature_extractor = FeatureExtractor(config)

    feature_cache_path = (
        Path(config.get("output.features_dir"))
        / f"{config.experiment_name}_features.npz"
    )

    if feature_cache_path.exists() and not FORCE_RECOMPUTE_FEATURES:
        print(f"Loading cached features from: {feature_cache_path}")
        cached = np.load(feature_cache_path, allow_pickle=True)
        X = cached["X"]
        y = cached["y"]
        groups = cached["groups"]
        skipped = []
    else:
        print("Extracting features from dataset...")
        X, y, groups, skipped = feature_extractor.extract_features_from_dataset(
            df_valid,
            show_progress=True
        )

        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(feature_cache_path, X=X, y=y, groups=groups)
        print(f"Saved features to: {feature_cache_path}")

    feature_names = feature_extractor.get_feature_names()

    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"Jumlah fitur X ({X.shape[1]}) tidak sama dengan feature_names ({len(feature_names)})."
        )

    return X, y, groups, feature_names, skipped


def build_feature_dataframe(df_valid, X, y, groups, feature_names):
    df_meta = df_valid.copy().reset_index(drop=True)

    if len(df_meta) != len(X):
        print("WARNING: jumlah df_valid berbeda dengan jumlah X.")
        print("Jika ada skipped samples, metadata dapat tidak sejajar.")
        print("Disarankan FORCE_RECOMPUTE_FEATURES=False jika cache dari eksperimen final sudah valid.")

    n = min(len(df_meta), len(X))

    df_meta = df_meta.iloc[:n].copy().reset_index(drop=True)
    X = X[:n]
    y = y[:n]
    groups = groups[:n]

    df_features = pd.DataFrame(X, columns=feature_names)

    df = pd.concat([df_meta, df_features], axis=1)
    df["sample_index"] = np.arange(len(df))
    df["persen_sisa"] = y
    df["group_feature"] = groups

    return df


def get_feature_columns(df):
    hsv_cols = [
        c for c in df.columns
        if c.startswith("before_hsv")
        or c.startswith("after_hsv")
        or c.startswith("delta_hsv")
    ]

    glcm_cols = [
        c for c in df.columns
        if "glcm" in c.lower()
    ]

    if len(hsv_cols) == 0:
        raise ValueError("Kolom HSV tidak ditemukan.")

    if len(glcm_cols) == 0:
        raise ValueError("Kolom GLCM tidak ditemukan.")

    print(f"Jumlah fitur HSV  : {len(hsv_cols)}")
    print(f"Jumlah fitur GLCM : {len(glcm_cols)}")

    return hsv_cols, glcm_cols


def find_counterexample_pairs(df, hsv_cols, glcm_cols):
    hsv_values = df[hsv_cols].to_numpy(dtype=float)
    glcm_values = df[glcm_cols].to_numpy(dtype=float)
    target_values = df["persen_sisa"].to_numpy(dtype=float)

    hsv_scaled = MinMaxScaler().fit_transform(hsv_values)
    glcm_scaled = MinMaxScaler().fit_transform(glcm_values)

    print("Menghitung pairwise distance HSV...")
    hsv_dist_matrix = pairwise_distances(hsv_scaled, metric="euclidean")

    print("Menghitung pairwise distance GLCM...")
    glcm_dist_matrix = pairwise_distances(glcm_scaled, metric="euclidean")

    print("Menghitung pairwise target difference...")
    target_diff_matrix = np.abs(target_values[:, None] - target_values[None, :])

    n = len(df)
    upper_i, upper_j = np.triu_indices(n, k=1)

    results_df = pd.DataFrame({
        "idx_A": upper_i,
        "idx_B": upper_j,
        "hsv_distance": hsv_dist_matrix[upper_i, upper_j],
        "glcm_distance": glcm_dist_matrix[upper_i, upper_j],
        "target_diff": target_diff_matrix[upper_i, upper_j],
        "target_A": target_values[upper_i],
        "target_B": target_values[upper_j],
    })

    hsv_threshold = results_df["hsv_distance"].quantile(HSV_QUANTILE)
    target_threshold = results_df["target_diff"].quantile(TARGET_QUANTILE)

    candidates = results_df[
        (results_df["hsv_distance"] <= hsv_threshold)
        & (results_df["target_diff"] >= target_threshold)
    ].copy()

    candidates["score"] = (
        candidates["glcm_distance"]
        * candidates["target_diff"]
        / (candidates["hsv_distance"] + 1e-8)
    )

    candidates = candidates.sort_values(
        by=["score", "glcm_distance", "target_diff"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("THRESHOLD PENCARIAN")
    print("=" * 100)
    print(f"HSV quantile       : {HSV_QUANTILE}")
    print(f"HSV threshold      : {hsv_threshold:.6f}")
    print(f"Target quantile    : {TARGET_QUANTILE}")
    print(f"Target threshold   : {target_threshold:.6f}")
    print(f"Jumlah kandidat    : {len(candidates)}")

    return candidates


def attach_metadata_to_candidates(candidates, df):
    rows = []

    for _, row in candidates.iterrows():
        idx_A = int(row["idx_A"])
        idx_B = int(row["idx_B"])

        meta_A = df.iloc[idx_A]
        meta_B = df.iloc[idx_B]

        rows.append({
            "idx_A": idx_A,
            "idx_B": idx_B,

            "food_A": meta_A["Name of the food"],
            "food_B": meta_B["Name of the food"],

            "target_A": row["target_A"],
            "target_B": row["target_B"],
            "target_diff": row["target_diff"],

            "hsv_distance": row["hsv_distance"],
            "glcm_distance": row["glcm_distance"],
            "score": row["score"],

            "before_A": meta_A["Image Before Eaten"],
            "after_A": meta_A["Image After Eaten"],
            "before_B": meta_B["Image Before Eaten"],
            "after_B": meta_B["Image After Eaten"],
        })

    return pd.DataFrame(rows)


def load_pair_images(config, row):
    preprocessor = ImagePreprocessor(config)

    img_before, img_after, mask_before, mask_after = preprocessor.process_image_pair(
        row["before_path"],
        row["after_path"]
    )

    if img_before is None or img_after is None:
        raise ValueError("Gagal membaca citra before/after.")

    if config.get("segmentation.enabled", False):
        img_before_used = preprocessor.apply_mask_to_image(img_before, mask_before)
        img_after_used = preprocessor.apply_mask_to_image(img_after, mask_after)
    else:
        img_before_used = img_before
        img_after_used = img_after

    return img_before, img_after, img_before_used, img_after_used


def compute_channel_histograms(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    h_hist, h_edges = np.histogram(h.flatten(), bins=HIST_BINS_H, range=(0, 180), density=True)
    s_hist, s_edges = np.histogram(s.flatten(), bins=HIST_BINS_S, range=(0, 256), density=True)
    v_hist, v_edges = np.histogram(v.flatten(), bins=HIST_BINS_V, range=(0, 256), density=True)

    h_centers = (h_edges[:-1] + h_edges[1:]) / 2
    s_centers = (s_edges[:-1] + s_edges[1:]) / 2
    v_centers = (v_edges[:-1] + v_edges[1:]) / 2

    return {
        "h": (h_centers, h_hist),
        "s": (s_centers, s_hist),
        "v": (v_centers, v_hist),
    }


def visualize_counterexample(config, df, candidates_meta, hsv_cols, glcm_cols, selected_order=0):
    if len(candidates_meta) == 0:
        raise ValueError("Tidak ada kandidat untuk divisualisasikan.")

    if selected_order >= len(candidates_meta):
        raise ValueError(
            f"SELECT_CANDIDATE_ORDER terlalu besar. Jumlah kandidat: {len(candidates_meta)}"
        )

    selected = candidates_meta.iloc[selected_order]

    idx_A = int(selected["idx_A"])
    idx_B = int(selected["idx_B"])

    row_A = df.iloc[idx_A]
    row_B = df.iloc[idx_B]

    img_A_before, img_A_after, img_A_before_used, img_A_after_used = load_pair_images(config, row_A)
    img_B_before, img_B_after, img_B_before_used, img_B_after_used = load_pair_images(config, row_B)

    hist_A_before = compute_channel_histograms(img_A_before_used)
    hist_A_after = compute_channel_histograms(img_A_after_used)
    hist_B_before = compute_channel_histograms(img_B_before_used)
    hist_B_after = compute_channel_histograms(img_B_after_used)

    glcm_A = row_A[glcm_cols].to_numpy(dtype=float)
    glcm_B = row_B[glcm_cols].to_numpy(dtype=float)

    glcm_feature_df = pd.DataFrame({
        "feature": glcm_cols,
        "sample_A": glcm_A,
        "sample_B": glcm_B,
        "abs_diff": np.abs(glcm_A - glcm_B)
    }).sort_values("abs_diff", ascending=False)

    top_glcm = glcm_feature_df.head(12).copy()

    fig = plt.figure(figsize=(20, 15))

    fig.suptitle(
        "Counterexample HSV Mirip, Target Berbeda, GLCM Berbeda\n"
        f"A: index {idx_A} | {row_A['Name of the food']} | sisa {row_A['persen_sisa']:.2f}%     "
        f"B: index {idx_B} | {row_B['Name of the food']} | sisa {row_B['persen_sisa']:.2f}%\n"
        f"HSV distance={selected['hsv_distance']:.4f} | "
        f"GLCM distance={selected['glcm_distance']:.4f} | "
        f"Target diff={selected['target_diff']:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    # ============================================================
    # ROW 1: RGB BEFORE-AFTER SAMPLE A DAN B
    # ============================================================

    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(img_A_before)
    ax1.set_title(f"A Before\n{row_A['Name of the food']}")
    ax1.axis("off")

    ax2 = plt.subplot(4, 4, 2)
    ax2.imshow(img_A_after)
    ax2.set_title(f"A After\nSisa {row_A['persen_sisa']:.2f}%")
    ax2.axis("off")

    ax3 = plt.subplot(4, 4, 3)
    ax3.imshow(img_B_before)
    ax3.set_title(f"B Before\n{row_B['Name of the food']}")
    ax3.axis("off")

    ax4 = plt.subplot(4, 4, 4)
    ax4.imshow(img_B_after)
    ax4.set_title(f"B After\nSisa {row_B['persen_sisa']:.2f}%")
    ax4.axis("off")

    # ============================================================
    # ROW 2: CITRA YANG DIPAKAI UNTUK FITUR
    # ============================================================

    ax5 = plt.subplot(4, 4, 5)
    ax5.imshow(img_A_before_used)
    ax5.set_title("A Before - Citra fitur")
    ax5.axis("off")

    ax6 = plt.subplot(4, 4, 6)
    ax6.imshow(img_A_after_used)
    ax6.set_title("A After - Citra fitur")
    ax6.axis("off")

    ax7 = plt.subplot(4, 4, 7)
    ax7.imshow(img_B_before_used)
    ax7.set_title("B Before - Citra fitur")
    ax7.axis("off")

    ax8 = plt.subplot(4, 4, 8)
    ax8.imshow(img_B_after_used)
    ax8.set_title("B After - Citra fitur")
    ax8.axis("off")

    # ============================================================
    # ROW 3: HISTOGRAM HSV A VS B
    # Menggunakan gabungan before+after secara visual: tampilkan after saja
    # agar tidak terlalu penuh.
    # ============================================================

    ax9 = plt.subplot(4, 4, 9)
    ax9.plot(hist_A_after["h"][0], hist_A_after["h"][1], label="A After")
    ax9.plot(hist_B_after["h"][0], hist_B_after["h"][1], label="B After")
    ax9.set_title("Histogram Hue After")
    ax9.set_xlabel("Hue")
    ax9.set_ylabel("Density")
    ax9.legend()
    ax9.grid(alpha=0.3)

    ax10 = plt.subplot(4, 4, 10)
    ax10.plot(hist_A_after["s"][0], hist_A_after["s"][1], label="A After")
    ax10.plot(hist_B_after["s"][0], hist_B_after["s"][1], label="B After")
    ax10.set_title("Histogram Saturation After")
    ax10.set_xlabel("Saturation")
    ax10.set_ylabel("Density")
    ax10.legend()
    ax10.grid(alpha=0.3)

    ax11 = plt.subplot(4, 4, 11)
    ax11.plot(hist_A_after["v"][0], hist_A_after["v"][1], label="A After")
    ax11.plot(hist_B_after["v"][0], hist_B_after["v"][1], label="B After")
    ax11.set_title("Histogram Value After")
    ax11.set_xlabel("Value")
    ax11.set_ylabel("Density")
    ax11.legend()
    ax11.grid(alpha=0.3)

    ax12 = plt.subplot(4, 4, 12)
    ax12.bar(["HSV distance", "GLCM distance"], [selected["hsv_distance"], selected["glcm_distance"]])
    ax12.set_title("Jarak Fitur A vs B")
    ax12.grid(axis="y", alpha=0.3)

    # ============================================================
    # ROW 4: TOP GLCM DIFFERENCE
    # ============================================================

    ax13 = plt.subplot(4, 1, 4)
    x = np.arange(len(top_glcm))
    width = 0.35

    ax13.bar(x - width / 2, top_glcm["sample_A"], width, label="Sample A")
    ax13.bar(x + width / 2, top_glcm["sample_B"], width, label="Sample B")

    ax13.set_xticks(x)
    ax13.set_xticklabels(top_glcm["feature"], rotation=30, ha="right")
    ax13.set_title("Top Perbedaan Nilai Fitur GLCM")
    ax13.set_ylabel("Nilai fitur")
    ax13.legend()
    ax13.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_VISUALIZATION:
        output_path = output_dir / f"counterexample_hsv_glcm_A{idx_A}_B{idx_B}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nGambar counterexample disimpan ke: {output_path}")

    plt.show()

    print("\n" + "=" * 100)
    print("COUNTEREXAMPLE TERPILIH")
    print("=" * 100)
    print(selected.to_string())

    print("\n" + "=" * 100)
    print("TOP PERBEDAAN GLCM")
    print("=" * 100)
    print(glcm_feature_df.head(20).to_string(index=False))

    glcm_path = Path(OUTPUT_DIR) / f"counterexample_glcm_features_A{idx_A}_B{idx_B}.csv"
    glcm_feature_df.to_csv(glcm_path, index=False)
    print(f"\nTabel perbedaan GLCM disimpan ke: {glcm_path}")

    return selected, glcm_feature_df


def main():
    config = cfg_module.load_config(CONFIG_PATH)

    print("=" * 100)
    print("MENCARI COUNTEREXAMPLE: HSV MIRIP, TARGET BERBEDA, GLCM BERBEDA")
    print("=" * 100)
    print(f"Config: {CONFIG_PATH}")
    print(f"Scenario: {config.get('experiment.scenario', 'unknown')}")
    print(f"Segmentation enabled: {config.get('segmentation.enabled', False)}")

    df_valid, _, _ = load_and_prepare_data(config)

    X, y, groups, feature_names, skipped = load_or_extract_features(config, df_valid)

    df = build_feature_dataframe(df_valid, X, y, groups, feature_names)

    hsv_cols, glcm_cols = get_feature_columns(df)

    candidates = find_counterexample_pairs(df, hsv_cols, glcm_cols)

    candidates_meta = attach_metadata_to_candidates(candidates, df)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"TOP {TOP_N_CANDIDATES} KANDIDAT COUNTEREXAMPLE")
    print("=" * 100)
    print(candidates_meta.head(TOP_N_CANDIDATES).to_string(index=False))

    if SAVE_CANDIDATES_CSV:
        candidates_path = output_dir / "counterexample_candidates.csv"
        candidates_meta.to_csv(candidates_path, index=False)
        print(f"\nCSV kandidat disimpan ke: {candidates_path}")

    visualize_counterexample(
        config=config,
        df=df,
        candidates_meta=candidates_meta,
        hsv_cols=hsv_cols,
        glcm_cols=glcm_cols,
        selected_order=SELECT_CANDIDATE_ORDER
    )


if __name__ == "__main__":
    main()