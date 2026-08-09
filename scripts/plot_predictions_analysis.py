import sys  # <--- WAJIB DITAMBAHKAN
from pathlib import Path
import pandas as pd

# 1. Atur path terlebih dahulu agar Python tahu di mana folder 'src' berada
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

# 2. Sekarang baru import modul lokal Anda
from utils.visualization import Visualizer

OUTPUT_DIR = Path("/home/rul/Skripsi/results/figures")

# Cari folder tempat script ini berada (Skripsi/scripts)
script_dir = Path(__file__).resolve().parent

# Gabungkan dengan nama file CSV
csv_path = script_dir / "prediction_best_model.csv"

# Load hasil prediksi model terbaik
df = pd.read_csv(csv_path)

# Inisialisasi visualizer (Gunakan OUTPUT_DIR yang sudah didefinisikan agar konsisten)
visualizer = Visualizer(save_dir=OUTPUT_DIR)

# Jalankan plot
visualizer.plot_prediction_scatter_food_analysis(
    df, filename="scatter_prediction_analysis.png"
)