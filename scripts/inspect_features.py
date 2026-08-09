import sys
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# PATH CONFIGURATION
# =====================================================
# Menambahkan folder 'src' ke sys.path agar Python mengenali folder 'features'
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

# Sekarang kita bisa mengimport dari package 'features'
from features.color_features import ColorFeatureExtractor

# Menentukan file sumber dan folder tujuan penyimpanan
FEATURE_FILE = "/home/rul/Skripsi/results/features/Raw EXP 1.1_features.npz"
OUTPUT_DIR = Path("/home/rul/Skripsi/results/metrics")

# Membuat folder 'metrics' otomatis jika belum ada
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD NPZ
# =====================================================
data = np.load(FEATURE_FILE)

X = data["X"]
y = data["y"]

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

# =====================================================
# FEATURE NAMES
# =====================================================
feature_names = []

# -----------------------------------------------------
# HSV
# Menggunakan nama fitur asli dari source code
# -----------------------------------------------------
color_extractor = ColorFeatureExtractor()
feature_names.extend(color_extractor.get_feature_names())

# -----------------------------------------------------
# GLCM
# sesuai texture_features.py
# -----------------------------------------------------
glcm_props = ["contrast", "homogeneity", "energy", "correlation"]

for prop in glcm_props:
    feature_names.append(f"{prop}_before")

for prop in glcm_props:
    feature_names.append(f"{prop}_after")

for prop in glcm_props:
    feature_names.append(f"{prop}_delta")

# -----------------------------------------------------
# AREA
# sesuai shape_features.py
# -----------------------------------------------------
feature_names.extend([
    "area_before",
    "area_after",
    "delta_area",
    "ratio_area"
])

# -----------------------------------------------------
# EDGE
# sesuai shape_features.py
# -----------------------------------------------------
feature_names.extend([
    "edge_density_before",
    "edge_density_after"
])

print(f"Feature count: {len(feature_names)}")

# =====================================================
# VALIDASI
# =====================================================
if len(feature_names) != X.shape[1]:
    raise ValueError(
        f"Jumlah feature name ({len(feature_names)}) "
        f"tidak sama dengan jumlah kolom X ({X.shape[1]})"
    )

# =====================================================
# DATAFRAME SELURUH FITUR
# =====================================================
df = pd.DataFrame(X, columns=feature_names)
df["target_leftover"] = y

# =====================================================
# SAVE CSV (Disimpan di results/metrics dengan nama sama seperti npz)
# =====================================================
# Mengambil nama file npz dan mengubah ekensinya menjadi .csv
npz_name = Path(FEATURE_FILE).with_suffix('.csv').name
output_csv = OUTPUT_DIR / npz_name

df.to_csv(output_csv, index=False)
print(f"\nSaved: {output_csv}")

# =====================================================
# TAMPILKAN FITUR AWAL DAN AKHIR
# =====================================================
print("\n=== 10 FITUR PERTAMA ===")
for f in feature_names[:10]:
    print(f)

print("\n=== 10 FITUR TERAKHIR ===")
for f in feature_names[-10:]:
    print(f)

# =====================================================
# SIMPAN SATU SAMPEL
# =====================================================
sample_idx = 0

sample_df = pd.DataFrame({
    "feature": feature_names,
    "value": X[sample_idx]
})

sample_csv = OUTPUT_DIR / f"sample_0_{npz_name}"

sample_df.to_csv(sample_csv, index=False)
print(f"\nSaved: {sample_csv}")