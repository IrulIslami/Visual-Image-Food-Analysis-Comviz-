# scripts/analyze_hsv_glcm_distance_space.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

# ============================================================
# EDIT BAGIAN INI SAJA
# ============================================================

CONFIG_PATH = "config/config_raw.yaml"

FORCE_RECOMPUTE_FEATURES = False

OUTPUT_DIR = "results/figures/hsv_glcm_distance_space"

# Pasangan yang ingin ditandai pada scatter plot
HIGHLIGHT_PAIRS = [
    (433, 434),  # Tahu bumbu kecap 96.55% vs 0.00%
    (429, 434),  # Tahu bumbu kecap 80.65% vs 0.00%
    (489, 490),  # Orek tempe 96.55% vs 0.00%
    (341, 342),  # Sambel Goreng Rempela Ati 0.00% vs 100.00%
]

# Untuk membuat scatter tidak terlalu berat kalau pasangan sangat banyak
# None = plot semua pasangan
MAX_SCATTER_POINTS = None

# ============================================================


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config as cfg_module
from data.loader import load_and_prepare_data
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


def compute_all_pair_distances(df, hsv_cols, glcm_cols):
    hsv_values = df[hsv_cols].to_numpy(dtype=float)
    glcm_values = df[glcm_cols].to_numpy(dtype=float)
    target_values = df["persen_sisa"].to_numpy(dtype=float)

    print("Scaling HSV features...")
    hsv_scaled = MinMaxScaler().fit_transform(hsv_values)

    print("Scaling GLCM features...")
    glcm_scaled = MinMaxScaler().fit_transform(glcm_values)

    print("Computing HSV pairwise distances...")
    hsv_dist_matrix = pairwise_distances(hsv_scaled, metric="euclidean")

    print("Computing GLCM pairwise distances...")
    glcm_dist_matrix = pairwise_distances(glcm_scaled, metric="euclidean")

    print("Computing target differences...")
    target_diff_matrix = np.abs(target_values[:, None] - target_values[None, :])

    n = len(df)
    idx_A, idx_B = np.triu_indices(n, k=1)

    all_pairs = pd.DataFrame({
        "idx_A": idx_A,
        "idx_B": idx_B,
        "hsv_distance": hsv_dist_matrix[idx_A, idx_B],
        "glcm_distance": glcm_dist_matrix[idx_A, idx_B],
        "target_A": target_values[idx_A],
        "target_B": target_values[idx_B],
        "target_diff": target_diff_matrix[idx_A, idx_B],
    })

    all_pairs["food_A"] = df.iloc[idx_A]["Name of the food"].to_numpy()
    all_pairs["food_B"] = df.iloc[idx_B]["Name of the food"].to_numpy()

    all_pairs["same_food"] = (
        all_pairs["food_A"].astype(str).str.lower().str.strip()
        == all_pairs["food_B"].astype(str).str.lower().str.strip()
    )

    all_pairs["before_A"] = df.iloc[idx_A]["Image Before Eaten"].to_numpy()
    all_pairs["after_A"] = df.iloc[idx_A]["Image After Eaten"].to_numpy()
    all_pairs["before_B"] = df.iloc[idx_B]["Image Before Eaten"].to_numpy()
    all_pairs["after_B"] = df.iloc[idx_B]["Image After Eaten"].to_numpy()

    # Percentile: posisi relatif terhadap semua pasangan
    all_pairs["hsv_percentile"] = all_pairs["hsv_distance"].rank(pct=True) * 100
    all_pairs["glcm_percentile"] = all_pairs["glcm_distance"].rank(pct=True) * 100
    all_pairs["target_percentile"] = all_pairs["target_diff"].rank(pct=True) * 100

    # Score counterexample:
    # tinggi jika target jauh, HSV dekat, GLCM jauh
    all_pairs["counterexample_score"] = (
        all_pairs["target_diff"]
        * all_pairs["glcm_percentile"]
        / (all_pairs["hsv_percentile"] + 1e-8)
    )

    return all_pairs


def print_distance_baseline(all_pairs):
    print("\n" + "=" * 100)
    print("BASELINE DISTRIBUSI DISTANCE UNTUK SEMUA PASANGAN")
    print("=" * 100)

    print("\nHSV distance describe:")
    print(all_pairs["hsv_distance"].describe())

    print("\nGLCM distance describe:")
    print(all_pairs["glcm_distance"].describe())

    print("\nTarget diff describe:")
    print(all_pairs["target_diff"].describe())

    print("\nQuantile HSV distance:")
    print(all_pairs["hsv_distance"].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

    print("\nQuantile GLCM distance:")
    print(all_pairs["glcm_distance"].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

    print("\nQuantile Target diff:")
    print(all_pairs["target_diff"].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))


def get_highlight_rows(all_pairs, highlight_pairs):
    rows = []

    for a, b in highlight_pairs:
        match = all_pairs[
            ((all_pairs["idx_A"] == a) & (all_pairs["idx_B"] == b))
            | ((all_pairs["idx_A"] == b) & (all_pairs["idx_B"] == a))
        ].copy()

        if len(match) == 0:
            print(f"WARNING: pasangan {a}-{b} tidak ditemukan.")
            continue

        match["highlight_label"] = f"A{a}-B{b}"
        rows.append(match.iloc[0])

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def print_highlight_summary(highlight_df):
    if len(highlight_df) == 0:
        return

    cols = [
        "highlight_label",
        "idx_A",
        "idx_B",
        "food_A",
        "food_B",
        "target_A",
        "target_B",
        "target_diff",
        "hsv_distance",
        "hsv_percentile",
        "glcm_distance",
        "glcm_percentile",
        "target_percentile",
        "same_food",
    ]

    print("\n" + "=" * 100)
    print("RINGKASAN PASANGAN YANG DI-HIGHLIGHT")
    print("=" * 100)
    print(highlight_df[cols].to_string(index=False))


def save_key_candidate_tables(all_pairs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Semua pasangan
    all_pairs_path = output_dir / "all_pair_distances.csv"
    all_pairs.to_csv(all_pairs_path, index=False)
    print(f"\nCSV semua pasangan disimpan ke: {all_pairs_path}")

    # Pasangan makanan sama, target jauh, HSV dekat, GLCM relatif tinggi
    same_food_candidates = all_pairs[
        (all_pairs["same_food"])
        & (all_pairs["target_percentile"] >= 75)
        & (all_pairs["hsv_percentile"] <= 10)
    ].copy()

    same_food_candidates = same_food_candidates.sort_values(
        by=["counterexample_score", "glcm_percentile", "target_diff"],
        ascending=[False, False, False]
    )

    same_food_path = output_dir / "same_food_counterexample_candidates.csv"
    same_food_candidates.to_csv(same_food_path, index=False)
    print(f"CSV kandidat same-food counterexample disimpan ke: {same_food_path}")

    # Baseline pasangan yang memang mirip targetnya
    # Berguna untuk menjawab: HSV distance untuk pasangan yang targetnya mirip berapa?
    small_target_diff = all_pairs[
        all_pairs["target_diff"] <= all_pairs["target_diff"].quantile(0.10)
    ].copy()

    small_target_diff_path = output_dir / "baseline_small_target_diff_pairs.csv"
    small_target_diff.to_csv(small_target_diff_path, index=False)
    print(f"CSV baseline target-diff kecil disimpan ke: {small_target_diff_path}")

    return all_pairs_path, same_food_path, small_target_diff_path


def plot_scatter_hsv_target(all_pairs, highlight_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if MAX_SCATTER_POINTS is not None and len(all_pairs) > MAX_SCATTER_POINTS:
        plot_df = all_pairs.sample(MAX_SCATTER_POINTS, random_state=42).copy()
    else:
        plot_df = all_pairs.copy()

    fig, ax = plt.subplots(figsize=(11, 7))

    sc = ax.scatter(
        plot_df["hsv_distance"],
        plot_df["target_diff"],
        c=plot_df["glcm_distance"],
        cmap="viridis",
        alpha=0.35,
        s=18
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("GLCM distance")

    if len(highlight_df) > 0:
        for _, row in highlight_df.iterrows():
            ax.scatter(
                row["hsv_distance"],
                row["target_diff"],
                color="red",
                s=250,
                marker="*",
                edgecolor="black",
                linewidth=1.2,
                zorder=5
            )

            ax.annotate(
                row["highlight_label"],
                xy=(row["hsv_distance"], row["target_diff"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="red"
            )

    hsv_q10 = all_pairs["hsv_distance"].quantile(0.10)
    target_q75 = all_pairs["target_diff"].quantile(0.75)

    ax.axvline(
        hsv_q10,
        linestyle="--",
        linewidth=1.5,
        label=f"HSV q10 = {hsv_q10:.3f}"
    )

    ax.axhline(
        target_q75,
        linestyle="--",
        linewidth=1.5,
        label=f"Target diff q75 = {target_q75:.2f}%"
    )

    ax.set_xlabel("HSV distance (lebih rendah = warna lebih mirip)")
    ax.set_ylabel("Target difference (%)")
    ax.set_title(
        "Ruang HSV Distance vs Target Difference\n"
        "Warna titik menunjukkan GLCM distance"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    output_path = output_dir / "scatter_hsv_distance_vs_target_diff.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Scatter plot disimpan ke: {output_path}")

    plt.show()


def plot_scatter_percentile(all_pairs, highlight_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if MAX_SCATTER_POINTS is not None and len(all_pairs) > MAX_SCATTER_POINTS:
        plot_df = all_pairs.sample(MAX_SCATTER_POINTS, random_state=42).copy()
    else:
        plot_df = all_pairs.copy()

    fig, ax = plt.subplots(figsize=(11, 7))

    sc = ax.scatter(
        plot_df["hsv_percentile"],
        plot_df["target_percentile"],
        c=plot_df["glcm_percentile"],
        cmap="viridis",
        alpha=0.35,
        s=18
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("GLCM percentile")

    if len(highlight_df) > 0:
        for _, row in highlight_df.iterrows():
            ax.scatter(
                row["hsv_percentile"],
                row["target_percentile"],
                color="red",
                s=250,
                marker="*",
                edgecolor="black",
                linewidth=1.2,
                zorder=5
            )

            ax.annotate(
                row["highlight_label"],
                xy=(row["hsv_percentile"], row["target_percentile"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="red"
            )

    ax.axvline(
        10,
        linestyle="--",
        linewidth=1.5,
        label="HSV percentile 10%"
    )

    ax.axhline(
        75,
        linestyle="--",
        linewidth=1.5,
        label="Target percentile 75%"
    )

    ax.set_xlabel("HSV percentile (lebih rendah = lebih mirip secara warna)")
    ax.set_ylabel("Target diff percentile (lebih tinggi = beda target lebih besar)")
    ax.set_title(
        "Percentile HSV vs Target Difference\n"
        "Warna titik menunjukkan posisi relatif GLCM"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    output_path = output_dir / "scatter_percentile_hsv_vs_target_glcm.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Percentile scatter plot disimpan ke: {output_path}")

    plt.show()


def plot_baseline_boxplot(all_pairs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = all_pairs.copy()

    df_plot["target_diff_group"] = pd.cut(
        df_plot["target_diff"],
        bins=[-0.01, 10, 25, 50, 75, 100],
        labels=[
            "0-10%",
            "10-25%",
            "25-50%",
            "50-75%",
            "75-100%"
        ]
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    groups = []
    labels = []

    for label in ["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"]:
        values = df_plot[df_plot["target_diff_group"] == label]["hsv_distance"].dropna()
        if len(values) > 0:
            groups.append(values)
            labels.append(label)

    ax.boxplot(groups, labels=labels, showfliers=False)

    ax.set_xlabel("Kelompok target difference")
    ax.set_ylabel("HSV distance")
    ax.set_title(
        "Baseline HSV Distance berdasarkan Selisih Persentase Sisa\n"
        "Dipakai untuk melihat skala 'mirip' dan 'tidak mirip' warna"
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "boxplot_hsv_distance_by_target_diff_group.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Boxplot baseline disimpan ke: {output_path}")

    plt.show()


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ANALISIS RUANG DISTANCE HSV, GLCM, DAN TARGET")
    print("=" * 100)

    config = cfg_module.load_config(CONFIG_PATH)

    print(f"Config               : {CONFIG_PATH}")
    print(f"Experiment scenario  : {config.get('experiment.scenario', 'unknown')}")
    print(f"Segmentation enabled : {config.get('segmentation.enabled', False)}")

    df_valid, _, _ = load_and_prepare_data(config)

    X, y, groups, feature_names, skipped = load_or_extract_features(config, df_valid)

    df = build_feature_dataframe(df_valid, X, y, groups, feature_names)

    hsv_cols, glcm_cols = get_feature_columns(df)

    all_pairs = compute_all_pair_distances(df, hsv_cols, glcm_cols)

    print_distance_baseline(all_pairs)

    highlight_df = get_highlight_rows(all_pairs, HIGHLIGHT_PAIRS)

    print_highlight_summary(highlight_df)

    save_key_candidate_tables(all_pairs, output_dir)

    highlight_path = output_dir / "highlight_pairs_summary.csv"
    highlight_df.to_csv(highlight_path, index=False)
    print(f"CSV highlight pairs disimpan ke: {highlight_path}")

    plot_scatter_hsv_target(all_pairs, highlight_df, output_dir)

    plot_scatter_percentile(all_pairs, highlight_df, output_dir)

    plot_baseline_boxplot(all_pairs, output_dir)

    print("\nSelesai.")


if __name__ == "__main__":
    main()