# scripts/visualize_glcm_pair.py

import sys
from pathlib import Path
from matplotlib.colors import LogNorm
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# ============================================================
# EDIT BAGIAN INI SAJA
# ============================================================

CONFIG_PATH = "config/config_raw.yaml"

# Pilihan mode:
SEARCH_MODE = "index"

# Untuk mode "index"
SAMPLE_INDEX = 0

# Untuk mode "filename"
TARGET_BEFORE_FILENAME = "029_413_DSC_0364_bef.JPG"

# Untuk mode "food_percent"
FOOD_KEYWORD = "tahu"
MIN_LEFTOVER = 30
MAX_LEFTOVER = 80
CANDIDATE_ORDER = 0

# Untuk mode "auto_low" atau "auto_high"
AVOID_FOODS = ["Bubur", "Nasi", "Tim"]
AUTO_CANDIDATE_ORDER = 0

# ------------------------------------------------------------
# PERUBAHAN DI SINI: Membuat SAVE_PATH Otomatis & Dinamis
# ------------------------------------------------------------
# 1. Tentukan dulu apa "nama indeks" atau penanda berdasarkan mode yang dipilih
if SEARCH_MODE == "index":
    index_name = str(SAMPLE_INDEX)
elif SEARCH_MODE == "filename":
    # Mengambil nama file tanpa ekstensi .JPG
    index_name = TARGET_BEFORE_FILENAME.split('.')[0] 
elif SEARCH_MODE == "food_percent":
    index_name = f"{FOOD_KEYWORD}_{MIN_LEFTOVER}to{MAX_LEFTOVER}_order{CANDIDATE_ORDER}"
elif SEARCH_MODE in ["auto_low", "auto_high"]:
    index_name = f"{SEARCH_MODE}_order{AUTO_CANDIDATE_ORDER}"
else:
    index_name = "default"

# 2. Gabungkan ke dalam format: glcm_visual_selected._"nama index".png
SAVE_PATH = f"results/figures/glcm/glcm_visual_selected.__{index_name}.png"

SAVE_FEATURE_TABLE = True

# GLCM visualisasi
VIS_DISTANCE = 1
VIS_ANGLE = 0

# ============================================================


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config as cfg_module
from data.loader import load_and_prepare_data
from data.preprocessor import ImagePreprocessor


def prepare_glcm_input(img_rgb, glcm_size=(128, 128), levels=8):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, glcm_size, interpolation=cv2.INTER_AREA)

    quantized = np.floor(
        (gray_resized.astype(np.float32) / 256.0) * levels
    ).astype(np.uint8)

    quantized = np.clip(quantized, 0, levels - 1)

    return gray_resized, quantized


def compute_glcm_and_features(quantized, levels=8, distance=1, angle=0):
    glcm = graycomatrix(
        quantized,
        distances=[distance],
        angles=[angle],
        levels=levels,
        symmetric=True,
        normed=True
    )

    features = {
        "contrast": graycoprops(glcm, "contrast")[0, 0],
        "homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
        "energy": graycoprops(glcm, "energy")[0, 0],
        "correlation": graycoprops(glcm, "correlation")[0, 0],
    }

    glcm_matrix = glcm[:, :, 0, 0]

    return glcm_matrix, features


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


def print_candidates(df, candidates, title, max_rows=30):
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

        print_candidates(df, match, "HASIL PENCARIAN BERDASARKAN FILENAME")

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
            df,
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

        print_candidates(df, candidates, "KANDIDAT AUTO_LOW: SISA TERENDAH")

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

        print_candidates(df, candidates, "KANDIDAT AUTO_HIGH: SISA TERTINGGI")

        if len(candidates) == 0:
            raise ValueError("Tidak ada kandidat untuk auto_high.")

        if AUTO_CANDIDATE_ORDER >= len(candidates):
            raise ValueError(
                f"AUTO_CANDIDATE_ORDER terlalu besar. Jumlah kandidat: {len(candidates)}"
            )

        return int(candidates.index[AUTO_CANDIDATE_ORDER])

    raise ValueError(f"SEARCH_MODE tidak dikenal: {mode}")


def visualize_glcm_pair(config_path, search_mode="index", save_path=None):
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

    glcm_size = tuple(config.get("features.glcm_size", [128, 128]))
    levels = int(config.get("features.glcm_levels", 8))

    gray_before, q_before = prepare_glcm_input(img_before_used, glcm_size, levels)
    gray_after, q_after = prepare_glcm_input(img_after_used, glcm_size, levels)

    glcm_before, feat_before = compute_glcm_and_features(
        q_before,
        levels=levels,
        distance=VIS_DISTANCE,
        angle=VIS_ANGLE
    )

    glcm_after, feat_after = compute_glcm_and_features(
        q_after,
        levels=levels,
        distance=VIS_DISTANCE,
        angle=VIS_ANGLE
    )

    feature_names = ["contrast", "homogeneity", "energy", "correlation"]

    before_values = np.array([feat_before[name] for name in feature_names])
    after_values = np.array([feat_after[name] for name in feature_names])
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
    print(f"glcm size                 : {glcm_size}")
    print(f"glcm levels               : {levels}")
    print(f"visual distance           : {VIS_DISTANCE}")
    print(f"visual angle              : {VIS_ANGLE}")

    fig = plt.figure(figsize=(18, 12))

    fig.suptitle(
        f"Visualisasi Perubahan GLCM Before-After\n"
        f"Index: {sample_index} | Makanan: {food_name} | Aktual sisa: {y_percent:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img_before)
    ax1.set_title("Before - RGB")
    ax1.axis("off")

    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(img_after)
    ax2.set_title("After - RGB")
    ax2.axis("off")

    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(img_before_used)
    ax3.set_title("Before - Citra untuk GLCM")
    ax3.axis("off")

    ax4 = plt.subplot(3, 4, 4)
    ax4.imshow(img_after_used)
    ax4.set_title("After - Citra untuk GLCM")
    ax4.axis("off")

    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(gray_before, cmap="gray")
    ax5.set_title("Before - Grayscale")
    ax5.axis("off")

    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(gray_after, cmap="gray")
    ax6.set_title("After - Grayscale")
    ax6.axis("off")

    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.imshow(q_before, cmap="gray", vmin=0, vmax=levels - 1)
    ax7.set_title(f"Before - Quantized ({levels} level)")
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    ax8 = plt.subplot(3, 4, 8)
    im8 = ax8.imshow(q_after, cmap="gray", vmin=0, vmax=levels - 1)
    ax8.set_title(f"After - Quantized ({levels} level)")
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    # max_val = max(glcm_before.max(), glcm_after.max())
    
    ax9 = plt.subplot(3, 4, 9)
    im9 = ax9.imshow(glcm_before)    
    # im9 = ax9.imshow(glcm_before, cmap='viridis', norm=LogNorm(vmin=0.0001, vmax=max_val))    
    ax9.set_title("GLCM Before")
    ax9.set_xlabel("Nilai piksel tetangga")
    ax9.set_ylabel("Nilai piksel pusat")
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    ax10 = plt.subplot(3, 4, 10)
    im10 = ax10.imshow(glcm_before)    
    # im10 = ax10.imshow(glcm_after, cmap='viridis', norm=LogNorm(vmin=0.0001, vmax=max_val))    
    ax10.set_title("GLCM After")
    ax10.set_xlabel("Nilai piksel tetangga")
    ax10.set_ylabel("Nilai piksel pusat")
    plt.colorbar(im10, ax=ax10, fraction=0.046)

    ax11 = plt.subplot(3, 4, 11)
    x = np.arange(len(feature_names))
    width = 0.35
    ax11.bar(x - width / 2, before_values, width, label="Before")
    ax11.bar(x + width / 2, after_values, width, label="After")
    ax11.set_xticks(x)
    ax11.set_xticklabels(feature_names, rotation=30)
    ax11.set_title("Nilai Fitur GLCM")
    ax11.legend()
    ax11.grid(axis="y", alpha=0.3)

    ax12 = plt.subplot(3, 4, 12)
    ax12.bar(feature_names, delta_values)
    ax12.axhline(0, linewidth=1)
    ax12.set_title("Delta GLCM (After - Before)")
    ax12.tick_params(axis="x", rotation=30)
    ax12.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nGambar disimpan ke: {save_path}")

    plt.show()

    df_features = pd.DataFrame({
        "fitur": feature_names,
        "before": before_values,
        "after": after_values,
        "delta_after_minus_before": delta_values
    })

    print("\n" + "=" * 100)
    print("NILAI FITUR GLCM")
    print("=" * 100)
    print(df_features.to_string(index=False))

    if SAVE_FEATURE_TABLE and save_path is not None:
        table_path = Path(save_path).with_suffix(".csv")
        df_features.to_csv(table_path, index=False)
        print(f"\nTabel fitur disimpan ke: {table_path}")

    return df_features


if __name__ == "__main__":
    visualize_glcm_pair(
        config_path=CONFIG_PATH,
        search_mode=SEARCH_MODE,
        save_path=SAVE_PATH
    )