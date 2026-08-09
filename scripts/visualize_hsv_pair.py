# scripts/visualize_hsv_pair.py

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# EDIT BAGIAN INI SAJA
# ============================================================

CONFIG_PATH = "config/config_raw.yaml"

# Pilihan mode:
# 1. "index"          -> pilih berdasarkan sample_index
# 2. "filename"       -> pilih berdasarkan nama file before
# 3. "food_percent"   -> cari otomatis berdasarkan nama makanan dan rentang sisa
# 4. "auto_low"       -> cari otomatis sisa rendah, selain makanan yang dihindari
# 5. "auto_high"      -> cari otomatis sisa tinggi, selain makanan yang dihindari
SEARCH_MODE = "index"

# Untuk mode "index"
SAMPLE_INDEX = 0

# Untuk mode "filename"
TARGET_BEFORE_FILENAME = "033_473_DSC_0929_bef.JPG"

# Untuk mode "food_percent"
FOOD_KEYWORD = "tahu"
MIN_LEFTOVER = 30
MAX_LEFTOVER = 80
CANDIDATE_ORDER = 0

# Untuk mode "auto_low" atau "auto_high"
AVOID_FOODS = ["Bubur", "Nasi", "Tim"]
AUTO_CANDIDATE_ORDER = 0

# Output
SAVE_PATH = "results/figures/hsv/hsv_visual_selected.png"
SAVE_FEATURE_TABLE = True

# Jumlah bin histogram HSV
HSV_BINS = (8, 8, 4)

# ============================================================


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config as cfg_module
from data.loader import load_and_prepare_data
from data.preprocessor import ImagePreprocessor


def add_percentage_column(df):
    wb = df["Weight Before Eaten (g)"].astype(float)
    wa = df["Weight After Eaten (g)"].astype(float)

    df["y_percent_leftover"] = 100.0 * np.divide(
        wa,
        wb,
        out=np.zeros_like(wa, dtype=float),
        where=wb != 0
    )

    df["y_percent_leftover"] = df["y_percent_leftover"].clip(0, 100)

    return df


def print_candidates(candidates, title, max_rows=30):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if len(candidates) == 0:
        print("Tidak ada kandidat yang cocok.")
        return

    show_cols = [
        "Name of the food",
        "y_percent_leftover",
        "Weight Before Eaten (g)",
        "Weight After Eaten (g)",
        "Image Before Eaten",
        "Image After Eaten"
    ]

    candidates_show = candidates[show_cols].copy()
    candidates_show.insert(0, "sample_index", candidates.index)

    print(candidates_show.head(max_rows).to_string(index=False))


def find_sample_index(df, mode):
    df = add_percentage_column(df.copy())

    if mode == "index":
        return SAMPLE_INDEX

    if mode == "filename":
        target = TARGET_BEFORE_FILENAME.lower().strip()

        match = df[
            df["Image Before Eaten"].astype(str).str.lower().str.strip() == target
        ]

        print_candidates(match, "HASIL PENCARIAN BERDASARKAN FILENAME")

        if len(match) == 0:
            raise ValueError(f"Filename before tidak ditemukan: {TARGET_BEFORE_FILENAME}")

        return int(match.index[0])

    if mode == "food_percent":
        keyword = FOOD_KEYWORD.lower().strip()

        candidates = df[
            df["Name of the food"].astype(str).str.lower().str.contains(keyword, na=False)
            & (df["y_percent_leftover"] >= MIN_LEFTOVER)
            & (df["y_percent_leftover"] <= MAX_LEFTOVER)
        ].copy()

        candidates = candidates.sort_values(
            by=["y_percent_leftover", "Name of the food"],
            ascending=[True, True]
        )

        print_candidates(
            candidates,
            f"KANDIDAT MAKANAN: keyword='{FOOD_KEYWORD}', sisa={MIN_LEFTOVER}-{MAX_LEFTOVER}%"
        )

        if len(candidates) == 0:
            raise ValueError("Tidak ada kandidat yang cocok untuk food_percent.")

        if CANDIDATE_ORDER >= len(candidates):
            raise ValueError(
                f"CANDIDATE_ORDER terlalu besar. Jumlah kandidat: {len(candidates)}"
            )

        return int(candidates.index[CANDIDATE_ORDER])

    if mode == "auto_low":
        avoid_pattern = "|".join([x.lower() for x in AVOID_FOODS])

        candidates = df[
            ~df["Name of the food"].astype(str).str.lower().str.contains(avoid_pattern, na=False)
        ].copy()

        candidates = candidates.sort_values(
            by=["y_percent_leftover", "Name of the food"],
            ascending=[True, True]
        )

        print_candidates(candidates, "KANDIDAT AUTO_LOW: SISA TERENDAH")

        if len(candidates) == 0:
            raise ValueError("Tidak ada kandidat untuk auto_low.")

        if AUTO_CANDIDATE_ORDER >= len(candidates):
            raise ValueError(
                f"AUTO_CANDIDATE_ORDER terlalu besar. Jumlah kandidat: {len(candidates)}"
            )

        return int(candidates.index[AUTO_CANDIDATE_ORDER])

    if mode == "auto_high":
        avoid_pattern = "|".join([x.lower() for x in AVOID_FOODS])

        candidates = df[
            ~df["Name of the food"].astype(str).str.lower().str.contains(avoid_pattern, na=False)
        ].copy()

        candidates = candidates.sort_values(
            by=["y_percent_leftover", "Name of the food"],
            ascending=[False, True]
        )

        print_candidates(candidates, "KANDIDAT AUTO_HIGH: SISA TERTINGGI")

        if len(candidates) == 0:
            raise ValueError("Tidak ada kandidat untuk auto_high.")

        if AUTO_CANDIDATE_ORDER >= len(candidates):
            raise ValueError(
                f"AUTO_CANDIDATE_ORDER terlalu besar. Jumlah kandidat: {len(candidates)}"
            )

        return int(candidates.index[AUTO_CANDIDATE_ORDER])

    raise ValueError(f"SEARCH_MODE tidak dikenal: {mode}")


def rgb_to_hsv_channels(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    return hsv, h, s, v


def compute_hsv_histogram(img_rgb, bins=(8, 8, 4)):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256]
    )

    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-8)

    return hist


def compute_channel_histograms(img_rgb):
    hsv, h, s, v = rgb_to_hsv_channels(img_rgb)

    h_hist, h_bins = np.histogram(h.flatten(), bins=36, range=(0, 180), density=True)
    s_hist, s_bins = np.histogram(s.flatten(), bins=32, range=(0, 256), density=True)
    v_hist, v_bins = np.histogram(v.flatten(), bins=32, range=(0, 256), density=True)

    h_centers = (h_bins[:-1] + h_bins[1:]) / 2
    s_centers = (s_bins[:-1] + s_bins[1:]) / 2
    v_centers = (v_bins[:-1] + v_bins[1:]) / 2

    return {
        "h": (h_centers, h_hist),
        "s": (s_centers, s_hist),
        "v": (v_centers, v_hist),
    }


def summarize_hsv(img_rgb):
    hsv, h, s, v = rgb_to_hsv_channels(img_rgb)

    return {
        "h_mean": float(np.mean(h)),
        "h_std": float(np.std(h)),
        "s_mean": float(np.mean(s)),
        "s_std": float(np.std(s)),
        "v_mean": float(np.mean(v)),
        "v_std": float(np.std(v)),
    }


def visualize_hsv_pair(config_path, search_mode="index", save_path=None):
    config = cfg_module.load_config(config_path)

    df_valid, _, _ = load_and_prepare_data(config)
    df_valid = add_percentage_column(df_valid)

    sample_index = find_sample_index(df_valid, search_mode)

    if sample_index >= len(df_valid):
        raise ValueError(f"sample_index terlalu besar. Jumlah data valid: {len(df_valid)}")

    row = df_valid.iloc[sample_index]

    preprocessor = ImagePreprocessor(config)

    img_before, img_after, mask_before, mask_after = preprocessor.process_image_pair(
        row["before_path"],
        row["after_path"]
    )

    if img_before is None or img_after is None:
        raise ValueError("Citra before atau after gagal diproses.")

    if config.get("segmentation.enabled", False):
        img_before_used = preprocessor.apply_mask_to_image(img_before, mask_before)
        img_after_used = preprocessor.apply_mask_to_image(img_after, mask_after)
    else:
        img_before_used = img_before
        img_after_used = img_after

    hsv_before, h_before, s_before, v_before = rgb_to_hsv_channels(img_before_used)
    hsv_after, h_after, s_after, v_after = rgb_to_hsv_channels(img_after_used)

    hist_before = compute_hsv_histogram(img_before_used, bins=HSV_BINS)
    hist_after = compute_hsv_histogram(img_after_used, bins=HSV_BINS)
    hist_delta = hist_after - hist_before

    channel_hist_before = compute_channel_histograms(img_before_used)
    channel_hist_after = compute_channel_histograms(img_after_used)

    summary_before = summarize_hsv(img_before_used)
    summary_after = summarize_hsv(img_after_used)

    feature_names = ["h_mean", "h_std", "s_mean", "s_std", "v_mean", "v_std"]
    before_values = np.array([summary_before[name] for name in feature_names])
    after_values = np.array([summary_after[name] for name in feature_names])
    delta_values = after_values - before_values

    food_name = row["Name of the food"]
    y_percent = row["y_percent_leftover"]

    print("\n" + "=" * 100)
    print("SAMPLE TERPILIH")
    print("=" * 100)
    print(f"sample_index              : {sample_index}")
    print(f"makanan                   : {food_name}")
    print(f"aktual sisa               : {y_percent:.2f}%")
    print(f"berat before              : {row['Weight Before Eaten (g)']}")
    print(f"berat after               : {row['Weight After Eaten (g)']}")
    print(f"before filename           : {row['Image Before Eaten']}")
    print(f"after filename            : {row['Image After Eaten']}")
    print(f"segmentation enabled      : {config.get('segmentation.enabled', False)}")
    print(f"hsv bins                  : {HSV_BINS}")

    fig = plt.figure(figsize=(18, 14))

    fig.suptitle(
        f"Visualisasi Perubahan HSV Before-After\n"
        f"Index: {sample_index} | Makanan: {food_name} | Aktual sisa: {y_percent:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(img_before)
    ax1.set_title("Before - RGB")
    ax1.axis("off")

    ax2 = plt.subplot(4, 4, 2)
    ax2.imshow(img_after)
    ax2.set_title("After - RGB")
    ax2.axis("off")

    ax3 = plt.subplot(4, 4, 3)
    ax3.imshow(img_before_used)
    ax3.set_title("Before - Citra untuk HSV")
    ax3.axis("off")

    ax4 = plt.subplot(4, 4, 4)
    ax4.imshow(img_after_used)
    ax4.set_title("After - Citra untuk HSV")
    ax4.axis("off")

    ax5 = plt.subplot(4, 4, 5)
    im5 = ax5.imshow(h_before, cmap="hsv", vmin=0, vmax=179)
    ax5.set_title("Before - Hue")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = plt.subplot(4, 4, 6)
    im6 = ax6.imshow(h_after, cmap="hsv", vmin=0, vmax=179)
    ax6.set_title("After - Hue")
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    ax7 = plt.subplot(4, 4, 7)
    im7 = ax7.imshow(s_before, cmap="gray", vmin=0, vmax=255)
    ax7.set_title("Before - Saturation")
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    ax8 = plt.subplot(4, 4, 8)
    im8 = ax8.imshow(s_after, cmap="gray", vmin=0, vmax=255)
    ax8.set_title("After - Saturation")
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    ax9 = plt.subplot(4, 4, 9)
    im9 = ax9.imshow(v_before, cmap="gray", vmin=0, vmax=255)
    ax9.set_title("Before - Value")
    ax9.axis("off")
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    ax10 = plt.subplot(4, 4, 10)
    im10 = ax10.imshow(v_after, cmap="gray", vmin=0, vmax=255)
    ax10.set_title("After - Value")
    ax10.axis("off")
    plt.colorbar(im10, ax=ax10, fraction=0.046)

    ax11 = plt.subplot(4, 4, 11)
    x_h_b, y_h_b = channel_hist_before["h"]
    x_h_a, y_h_a = channel_hist_after["h"]
    ax11.plot(x_h_b, y_h_b, label="Before")
    ax11.plot(x_h_a, y_h_a, label="After")
    ax11.set_title("Histogram Hue")
    ax11.set_xlabel("Hue")
    ax11.set_ylabel("Density")
    ax11.legend()
    ax11.grid(alpha=0.3)

    ax12 = plt.subplot(4, 4, 12)
    x_s_b, y_s_b = channel_hist_before["s"]
    x_s_a, y_s_a = channel_hist_after["s"]
    ax12.plot(x_s_b, y_s_b, label="Before")
    ax12.plot(x_s_a, y_s_a, label="After")
    ax12.set_title("Histogram Saturation")
    ax12.set_xlabel("Saturation")
    ax12.set_ylabel("Density")
    ax12.legend()
    ax12.grid(alpha=0.3)

    ax13 = plt.subplot(4, 4, 13)
    x_v_b, y_v_b = channel_hist_before["v"]
    x_v_a, y_v_a = channel_hist_after["v"]
    ax13.plot(x_v_b, y_v_b, label="Before")
    ax13.plot(x_v_a, y_v_a, label="After")
    ax13.set_title("Histogram Value")
    ax13.set_xlabel("Value")
    ax13.set_ylabel("Density")
    ax13.legend()
    ax13.grid(alpha=0.3)

    ax14 = plt.subplot(4, 4, 14)
    x = np.arange(len(feature_names))
    width = 0.35
    ax14.bar(x - width / 2, before_values, width, label="Before")
    ax14.bar(x + width / 2, after_values, width, label="After")
    ax14.set_xticks(x)
    ax14.set_xticklabels(feature_names, rotation=35)
    ax14.set_title("Ringkasan HSV")
    ax14.legend()
    ax14.grid(axis="y", alpha=0.3)

    ax15 = plt.subplot(4, 4, 15)
    ax15.bar(feature_names, delta_values)
    ax15.axhline(0, linewidth=1)
    ax15.set_title("Delta HSV (After - Before)")
    ax15.tick_params(axis="x", rotation=35)
    ax15.grid(axis="y", alpha=0.3)

    ax16 = plt.subplot(4, 4, 16)
    top_k = 20
    top_indices = np.argsort(np.abs(hist_delta))[-top_k:]
    top_indices = top_indices[np.argsort(np.abs(hist_delta[top_indices]))[::-1]]

    ax16.bar(range(top_k), hist_delta[top_indices])
    ax16.axhline(0, linewidth=1)
    ax16.set_title(f"Top {top_k} Delta HSV Histogram Bin")
    ax16.set_xlabel("Ranking bin")
    ax16.set_ylabel("Delta")
    ax16.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nGambar disimpan ke: {save_path}")

    plt.show()

    df_summary = pd.DataFrame({
        "fitur": feature_names,
        "before": before_values,
        "after": after_values,
        "delta_after_minus_before": delta_values
    })

    print("\n" + "=" * 100)
    print("RINGKASAN NILAI HSV")
    print("=" * 100)
    print(df_summary.to_string(index=False))

    df_top_delta = pd.DataFrame({
        "rank": np.arange(1, top_k + 1),
        "hist_bin_index": top_indices,
        "delta_after_minus_before": hist_delta[top_indices]
    })

    print("\n" + "=" * 100)
    print(f"TOP {top_k} DELTA HSV HISTOGRAM BIN")
    print("=" * 100)
    print(df_top_delta.to_string(index=False))

    if SAVE_FEATURE_TABLE and save_path is not None:
        summary_path = Path(save_path).with_name(Path(save_path).stem + "_summary.csv")
        delta_path = Path(save_path).with_name(Path(save_path).stem + "_top_delta_bins.csv")

        df_summary.to_csv(summary_path, index=False)
        df_top_delta.to_csv(delta_path, index=False)

        print(f"\nTabel ringkasan HSV disimpan ke: {summary_path}")
        print(f"Tabel top delta bin HSV disimpan ke: {delta_path}")

    return df_summary, df_top_delta


if __name__ == "__main__":
    visualize_hsv_pair(
        config_path=CONFIG_PATH,
        search_mode=SEARCH_MODE,
        save_path=SAVE_PATH
    )