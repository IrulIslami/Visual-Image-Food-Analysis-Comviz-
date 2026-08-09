import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import cv2
from skimage.feature import graycomatrix, graycoprops


class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save figures (None = don't save)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        filename: str = "predictions.png"
    ):
        """Plot predicted vs actual values."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Leftover %', fontsize=12)
        ax.set_ylabel('Predicted Leftover %', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "residuals.png"
    ):
        """Plot residual distribution and residuals vs predicted."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residual histogram
        axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residuals', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Leftover %', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_scatter_food_analysis(
        self,
        df,
        filename="food_error_scatter.png"
    ):
        """
        Scatter plot aktual vs prediksi dengan highlight makanan 
        yang memiliki karakteristik visual sulit.
        """
        highlight_foods = {
            "Tim",
            "Bali Telur",
            "Opor Telur",
            "Telur Mata Sapi",
            "Rendang ayam"
        }

        fig, ax = plt.subplots(figsize=(8, 8))

        normal_df = df[
            ~df["Name of the food"].isin(highlight_foods)
        ]

        special_df = df[
            df["Name of the food"].isin(highlight_foods)
        ]

        # Seluruh data
        ax.scatter(
            normal_df["csv_y_true"],
            normal_df["csv_y_pred"],
            alpha=0.5,
            s=50,
            label="Makanan lainnya"
        )

        # Makanan yang dibahas
        ax.scatter(
            special_df["csv_y_true"],
            special_df["csv_y_pred"],
            s=100,
            label="Tekstur homogen / kasus sulit"
        )

        # Garis ideal
        ax.plot(
            [0, 100],
            [0, 100],
            "--",
            linewidth=2,
            label="y = x"
        )

        # Anotasi
        for _, row in special_df.iterrows():
            if row["csv_abs_error"] >= 20:
                ax.annotate(
                    row["Name of the food"],
                    (
                        row["csv_y_true"],
                        row["csv_y_pred"]
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )

        ax.set_xlabel("Persentase Sisa Makanan Aktual (%)")
        ax.set_ylabel("Persentase Sisa Makanan Prediksi (%)")
        ax.set_title("Scatter Plot Nilai Aktual dan Prediksi")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(
                self.save_dir / filename,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()
    
    def plot_feature_importance(
        self,
        feature_importance_df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "feature_importance.png"
    ):
        """Plot feature importance."""
        df_plot = feature_importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        ax.barh(range(len(df_plot)), df_plot['importance'], color='steelblue')
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_cv_results(
        self,
        cv_results_df: pd.DataFrame,
        filename: str = "cv_results.png"
    ):
        """Plot cross-validation results."""
        metrics = ['mae', 'mse', 'r2']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = cv_results_df[metric]
            
            ax.plot(cv_results_df['fold'], values, marker='o', linewidth=2, markersize=8)
            ax.axhline(values.mean(), color='r', linestyle='--', label=f'Mean: {values.mean():.3f}')
            ax.fill_between(
                cv_results_df['fold'],
                values.mean() - values.std(),
                values.mean() + values.std(),
                alpha=0.2, color='red'
            )
            
            ax.set_xlabel('Fold', fontsize=11)
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(f'{metric.upper()} Across Folds', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_cross_representation(
        self, 
        area_data: dict, 
        hsv_data: dict, 
        glcm_data: dict, 
        patch_size: int = 100, 
        filename: str = "Visualisasi_Pendukung_Feature_Importance.png"
    ):
        """Plot gabungan lintas fitur dan makanan terpilih dengan format (Gambar Kiri -> Penjelasan Kanan) dengan font diperbesar."""
        import matplotlib.gridspec as gridspec
        
        # Ukuran kanvas dilebarkan agar memuat font besar dengan aman
        fig = plt.figure(figsize=(20, 15))
        # Mengatur rasio lebar agar kolom teks (kolom 1 dan 3) mendapat ruang ekstra
        gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1.4, 1, 1.4])
        
        # =====================================================================
        # BARIS 1: Analisis Anatomi Area (Sampel: Rendang)
        # =====================================================================
        ax_area_imgb = fig.add_subplot(gs[0, 0])
        ax_area_txtb = fig.add_subplot(gs[0, 1])
        ax_area_imga = fig.add_subplot(gs[0, 2])
        ax_area_txta = fig.add_subplot(gs[0, 3])
        
        img_areab = cv2.imread(area_data['bef'])
        img_areaa = cv2.imread(area_data['aft'])
        
        gray_b = cv2.cvtColor(img_areab, cv2.COLOR_BGR2GRAY)
        gray_a = cv2.cvtColor(img_areaa, cv2.COLOR_BGR2GRAY)
        total_pixels = gray_b.shape[0] * gray_b.shape[1]
        
        fb = np.count_nonzero(gray_b)
        bg_b = total_pixels - fb
        rb = fb / total_pixels
        
        fa = np.count_nonzero(gray_a)
        bg_a = total_pixels - fa
        ra = fa / total_pixels
        
        # Kolom 0: Gambar Before
        ax_area_imgb.imshow(cv2.cvtColor(img_areab, cv2.COLOR_BGR2RGB))
        ax_area_imgb.set_title(f"1. Fitur Luas Area: {area_data['food']} (Bef)", fontweight='bold', fontsize=14)
        ax_area_imgb.axis('off')
        
        # Kolom 1: Penjelasan Before
        ax_area_txtb.axis('off')
        text_b = (f"Total Piksel Citra : {total_pixels:,}\n"
                  f"Piksel Latar (Hitam) : {bg_b:,}\n"
                  f"Piksel Makanan       : {fb:,}\n\n"
                  f"Rasio Area = {fb:,} / {total_pixels:,}\n"
                  f"Hasil Fitur = {rb:.4f} ({rb*100:.2f}%)")
        ax_area_txtb.text(0.05, 0.5, text_b, ha='left', va='center', fontsize=14, 
                          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.8'))
        
        # Kolom 2: Gambar After
        ax_area_imga.imshow(cv2.cvtColor(img_areaa, cv2.COLOR_BGR2RGB))
        ax_area_imga.set_title(f"1. Fitur Luas Area: {area_data['food']} (Aft)", fontweight='bold', fontsize=14)
        ax_area_imga.axis('off')
        
        # Kolom 3: Penjelasan After
        ax_area_txta.axis('off')
        text_a = (f"Total Piksel Citra : {total_pixels:,}\n"
                  f"Piksel Latar (Hitam) : {bg_a:,}\n"
                  f"Piksel Sisa Makanan  : {fa:,}\n\n"
                  f"Rasio Area = {fa:,} / {total_pixels:,}\n"
                  f"Hasil Fitur = {ra:.4f} ({ra*100:.2f}%)")
        ax_area_txta.text(0.05, 0.5, text_a, ha='left', va='center', fontsize=14, 
                          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.8'))
        
        # =====================================================================
        # BARIS 2: Analisis Pergeseran Warna/HSV (Sampel: Nasi)
        # =====================================================================
        ax_hsv_imgb = fig.add_subplot(gs[1, 0])
        ax_hsv_histb = fig.add_subplot(gs[1, 1])
        ax_hsv_imga = fig.add_subplot(gs[1, 2])
        ax_hsv_hista = fig.add_subplot(gs[1, 3])
        
        img_hsvb = cv2.imread(hsv_data['bef'])
        img_hsva = cv2.imread(hsv_data['aft'])
        hb = cv2.cvtColor(img_hsvb, cv2.COLOR_BGR2HSV)[:,:,0]
        ha = cv2.cvtColor(img_hsva, cv2.COLOR_BGR2HSV)[:,:,0]
        mb = (hb > 0).astype(np.uint8)
        ma = (ha > 0).astype(np.uint8)
        
        # Kolom 0 & 1: Before
        ax_hsv_imgb.imshow(cv2.cvtColor(img_hsvb, cv2.COLOR_BGR2RGB))
        ax_hsv_imgb.set_title(f"2. Fitur Warna: {hsv_data['food']} (Bef)", fontsize=14, fontweight='bold')
        ax_hsv_imgb.axis('off')
        
        ax_hsv_histb.hist(hb[mb == 1].ravel(), bins=180, range=[1, 180], color='orange', alpha=0.8)
        ax_hsv_histb.set_title("Histogram Hue (Before)", fontsize=14)
        ax_hsv_histb.set_ylabel("Frekuensi Piksel", fontsize=12)
        ax_hsv_histb.set_xlabel("Nilai Hue", fontsize=12)
        ax_hsv_histb.tick_params(labelsize=11)
        ax_hsv_histb.grid(True, alpha=0.3)
        
        # Kolom 2 & 3: After
        ax_hsv_imga.imshow(cv2.cvtColor(img_hsva, cv2.COLOR_BGR2RGB))
        ax_hsv_imga.set_title(f"2. Fitur Warna: {hsv_data['food']} (Aft)", fontsize=14, fontweight='bold')
        ax_hsv_imga.axis('off')
        
        ax_hsv_hista.hist(ha[ma == 1].ravel(), bins=180, range=[1, 180], color='blue', alpha=0.8)
        ax_hsv_hista.set_title("Histogram Hue (After)", fontsize=14)
        ax_hsv_hista.set_ylabel("Frekuensi Piksel", fontsize=12)
        ax_hsv_hista.set_xlabel("Nilai Hue", fontsize=12)
        ax_hsv_hista.tick_params(labelsize=11)
        ax_hsv_hista.grid(True, alpha=0.3)
        
        # =====================================================================
        # BARIS 3: Analisis Tekstur GLCM Kuantisasi 8 Level (Sampel: Bubur)
        # =====================================================================
        ax_glcm_imgb = fig.add_subplot(gs[2, 0])
        ax_glcm_txtb = fig.add_subplot(gs[2, 1])
        ax_glcm_imga = fig.add_subplot(gs[2, 2])
        ax_glcm_txta = fig.add_subplot(gs[2, 3])
        
        img_glcmb = cv2.imread(glcm_data['bef'], cv2.IMREAD_GRAYSCALE)
        img_glcma = cv2.imread(glcm_data['aft'], cv2.IMREAD_GRAYSCALE)
        
        h_gb, w_gb = img_glcmb.shape
        h_ga, w_ga = img_glcma.shape
        patch_b = img_glcmb[h_gb//2 : h_gb//2 + patch_size, w_gb//2 : w_gb//2 + patch_size]
        patch_a = img_glcma[h_ga//2 : h_ga//2 + patch_size, w_ga//2 : w_ga//2 + patch_size]
        
        # Kuantisasi patch menjadi 8 tingkat keabuan (nilai 0 - 7)
        patch_b_quant = (patch_b // 32).astype(np.uint8)
        patch_a_quant = (patch_a // 32).astype(np.uint8)
        
        # Hitung GLCM menggunakan patch yang sudah dikuantisasi dan set levels=8
        g_b = graycomatrix(patch_b_quant, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
        g_a = graycomatrix(patch_a_quant, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
        
        hom_b = graycoprops(g_b, 'homogeneity')[0, 0]
        con_b = graycoprops(g_b, 'contrast')[0, 0]
        hom_a = graycoprops(g_a, 'homogeneity')[0, 0]
        con_a = graycoprops(g_a, 'contrast')[0, 0]
        
        # Kolom 0 & 1: Before
        ax_glcm_imgb.imshow(patch_b, cmap='gray')
        ax_glcm_imgb.set_title(f"3. Fitur Tekstur: {glcm_data['food']} (Patch Sebelum)", fontweight='bold', fontsize=14)
        ax_glcm_imgb.axis('off')
        
        ax_glcm_txtb.axis('off')
        text_glcm_b = (f"Kondisi: Tekstur Makanan (Kasar)\n"
                       f"Kuantisasi GLCM: 8 Level\n\n"
                       f"Homogeneity : {hom_b:.4f}\n"
                       f"Contrast    : {con_b:.4f}")
        ax_glcm_txtb.text(0.05, 0.5, text_glcm_b, ha='left', va='center', fontsize=14, 
                          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.8'))
        
        # Kolom 2 & 3: After
        ax_glcm_imga.imshow(patch_a, cmap='gray')
        ax_glcm_imga.set_title(f"3. Fitur Tekstur: Piring Kosong (Patch Sesudah)", fontweight='bold', fontsize=14)
        ax_glcm_imga.axis('off')
        
        ax_glcm_txta.axis('off')
        text_glcm_a = (f"Kondisi: Permukaan Piring (Halus)\n"
                       f"Kuantisasi GLCM: 8 Level\n\n"
                       f"Homogeneity : {hom_a:.4f}\n"
                       f"Contrast    : {con_a:.4f}")
        ax_glcm_txta.text(0.05, 0.5, text_glcm_a, ha='left', va='center', fontsize=14, 
                          bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.8'))
        
        # Penyesuaian Layout Akhir
        plt.suptitle("Visualisasi Pendukung Analisis Ragam Karakteristik Fitur Eksperimen", 
                     fontsize=18, fontweight='bold', y=0.96)
        plt.tight_layout(pad=3.5, h_pad=3.5, w_pad=2.0)
        
        if self.save_dir:
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()