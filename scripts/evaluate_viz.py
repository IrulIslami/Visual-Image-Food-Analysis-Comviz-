# import os
# import sys
# import cv2
# from pathlib import Path

# # Daftarkan root path proyek
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.utils.visualization import Visualizer
# from src.data.preprocessor import ImagePreprocessor
# import src.config as cfg_module  # Untuk meload konfigurasi seperti run_experiment.py

# def main():
#     # 1. Load Konfigurasi (Pastikan menunjuk ke config segmented eksperimen Anda)
#     config_path = "config/config_segmented.yaml"
#     if not os.path.exists(config_path):
#         print(f"Error: File konfigurasi tidak ditemukan di {config_path}!")
#         return
        
#     config = cfg_module.load_config(config_path)

#     # 2. Inisialisasi Preprocessor dan Visualizer
#     # Preprocessor ini akan otomatis mengikuti ukuran citra & iterasi GrabCut dari config
#     preprocessor = ImagePreprocessor(config)
#     viz = Visualizer(save_dir="results/figures")

#     # Daftar nama file mentah Anda
#     bef_filenames = ["000_000_DSC_0016_bef.JPG", "001_064_DSC_0738_bef.JPG", "011_197_DSC_0441_bef.JPG"]
#     aft_filenames = ["000_000_DSC_0032_aft.JPG", "001_064_DSC_0790_aft.JPG", "011_197_DSC_0480_aft.JPG"]
#     food_names = ["Bubur", "Nasi", "Rendang"]

#     for n, bef_name in enumerate(bef_filenames):
#         aft_name = aft_filenames[n]
#         food = food_names[n]
        
#         print(f"\nMemproses visualisasi untuk sampel: {food} ({bef_name})")

#         folder_prefix = bef_name[:3]
        
#         # Bentuk path lengkap ke gambar raw
#         raw_before_path = Path(f"data/raw/before/{folder_prefix}/{bef_name}")
#         raw_after_path = Path(f"data/raw/after/{folder_prefix}/{aft_name}")
        
#         if not raw_before_path.exists() or not raw_after_path.exists():
#             print(f"  -> File mentah tidak ditemukan untuk {food}. Cek letak filenya!")
#             continue

#         # =========================================================================
#         # 3. LAKUKAN SEGMENTASI "ON THE FLY" MENGGUNAKAN CLASS ANDA
#         # =========================================================================
#         # Step A: Baca gambar, resize, normalisasi lighting, dan dapatkan mask binary GrabCut
#         img_before_rgb, img_after_rgb, mask_before, mask_after = preprocessor.process_image_pair(
#             raw_before_path, raw_after_path
#         )

#         if img_before_rgb is None or img_after_rgb is None:
#             print(f"  -> Gagal memproses gambar {food}.")
#             continue

#         # Step B: Terapkan mask ke gambar agar latar belakang menjadi nol (Hitam)
#         before_segmented_rgb = preprocessor.apply_mask_to_image(img_before_rgb, mask_before)
#         after_segmented_rgb = preprocessor.apply_mask_to_image(img_after_rgb, mask_after)

#         # Step C: Konversi KEMBALI ke BGR agar warna asli tidak kacau saat disimpan cv2.imwrite
#         img_before_seg_bgr = cv2.cvtColor(before_segmented_rgb, cv2.COLOR_RGB2BGR)
#         img_after_seg_bgr = cv2.cvtColor(after_segmented_rgb, cv2.COLOR_RGB2BGR)

#         # 4. Simpan gambar segmentasi secara sementara (Temporary Save)
#         temp_bef = f"results/figures/temp_bef_{n}.png"
#         temp_aft = f"results/figures/temp_aft_{n}.png"
#         cv2.imwrite(temp_bef, img_before_seg_bgr)
#         cv2.imwrite(temp_aft, img_after_seg_bgr)
        
#         # 5. Bangun Visualisasinya (yang membaca dari file temporary tersebut)
#         print(f"  -> Membangun grafik HSV dan GLCM untuk {food}...")
        
#         viz.plot_hsv_comparison(
#             before_path=temp_bef, 
#             after_path=temp_aft, 
#             filename=f"Visualisasi_{food}_HSV.png"
#         )
        
#         viz.plot_glcm_comparison(
#             before_path=temp_bef, 
#             after_path=temp_aft, 
#             patch_size=100,
#             filename=f"Visualisasi_{food}_GLCM.png"
#         )
        
#         viz.plot_area_anatomy(
#             before_path=temp_bef,
#             after_path=temp_aft,
#             filename=f"Visualisasi_{food}_area.png"
#         )
        
#         # 6. Bersihkan (Hapus) file gambar sementara
#         if os.path.exists(temp_bef): os.remove(temp_bef)
#         if os.path.exists(temp_aft): os.remove(temp_aft)
        
#     print("\n✅ Semua visualisasi berhasil dibuat dan tersimpan di folder results/figures!")

# if __name__ == "__main__":
#     main()

import os
import sys
import cv2
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.visualization import Visualizer
from src.data.preprocessor import ImagePreprocessor
import src.config as cfg_module

def main():
    config_path = "config/config_segmented.yaml"
    config = cfg_module.load_config(config_path)

    preprocessor = ImagePreprocessor(config)
    viz = Visualizer(save_dir="results/figures")

    # Mapping target sampel spesifik sesuai representasi terbaiknya
    targets = {
        "Bubur":   {"bef": "000_000_DSC_0016_bef.JPG", "aft": "000_000_DSC_0032_aft.JPG"},
        "Nasi":    {"bef": "001_064_DSC_0738_bef.JPG", "aft": "001_064_DSC_0790_aft.JPG"},
        "Rendang": {"bef": "011_197_DSC_0441_bef.JPG", "aft": "011_197_DSC_0480_aft.JPG"}
    }

    processed_paths = {}

    print("Memulai pemrosesan citra tersegmentasi secara on the fly...")
    for food, files in targets.items():
        folder_prefix = files['bef'][:3]
        raw_before_path = Path(f"data/raw/before/{folder_prefix}/{files['bef']}")
        raw_after_path = Path(f"data/raw/after/{folder_prefix}/{files['aft']}")
        
        if not raw_before_path.exists() or not raw_after_path.exists():
            print(f"Error: Gambar untuk {food} tidak ditemukan.")
            return

        # Jalankan GrabCut segmentasi
        img_before_rgb, img_after_rgb, mask_before, mask_after = preprocessor.process_image_pair(
            raw_before_path, raw_after_path
        )
        
        before_seg = preprocessor.apply_mask_to_image(img_before_rgb, mask_before)
        after_seg = preprocessor.apply_mask_to_image(img_after_rgb, mask_after)
        
        # Konversi balik ke BGR untuk penyimpanan cv2.imwrite
        before_bgr = cv2.cvtColor(before_seg, cv2.COLOR_RGB2BGR)
        after_bgr = cv2.cvtColor(after_seg, cv2.COLOR_RGB2BGR)
        
        temp_bef_path = f"results/figures/temp_{food}_bef.png"
        temp_aft_path = f"results/figures/temp_{food}_aft.png"
        
        cv2.imwrite(temp_bef_path, before_bgr)
        cv2.imwrite(temp_aft_path, after_bgr)
        
        processed_paths[food] = {
            "food": food,
            "bef": temp_bef_path,
            "aft": temp_aft_path
        }

    print("\nMembangun satu plot visual gabungan terpadu untuk Bab 6.3...")
    viz.plot_cross_representation(
        area_data=processed_paths["Rendang"],
        hsv_data=processed_paths["Nasi"],
        glcm_data=processed_paths["Bubur"],
        patch_size=100,
        filename="Visualisasi_Pendukung_Feature_Importance.png"
    )

    print("\nCleaning up file gambar sementara...")
    for food in targets.keys():
        if os.path.exists(processed_paths[food]["bef"]): os.remove(processed_paths[food]["bef"])
        if os.path.exists(processed_paths[food]["aft"]): os.remove(processed_paths[food]["aft"])
        
    print("✅ File tunggal 'Visualisasi_Pendukung_Feature_Importance.png' sukses dibuat di folder results/figures!")

if __name__ == "__main__":
    main()