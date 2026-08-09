import sys
from pathlib import Path

# Cari path ke folder 'src'
# __file__ adalah /home/rul/Skripsi/scripts/check.py
# .parent.parent adalah /home/rul/Skripsi
# lalu kita masuk ke /home/rul/Skripsi/src
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

# Sekarang kamu bisa import dengan format: from features.nama_file import ...
from features.color_features import ColorFeatureExtractor
from features.texture_features import TextureFeatureExtractor
from features.shape_features import ShapeFeatureExtractor

print(hasattr(ColorFeatureExtractor(), "get_feature_names"))
print(hasattr(TextureFeatureExtractor(), "get_feature_names"))
print(hasattr(ShapeFeatureExtractor(), "get_feature_names"))