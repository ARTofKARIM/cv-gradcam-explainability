"""Visualization for Grad-CAM heatmaps."""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class GradCAMVisualizer:
    def __init__(self, output_dir="results/", alpha=0.4):
        self.output_dir = output_dir
        self.alpha = alpha

    def overlay_heatmap(self, image, heatmap, colormap="jet"):
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_color = plt.cm.get_cmap(colormap)(heatmap_resized)[:, :, :3]
        if image.max() > 1:
            image_norm = image / 255.0
        else:
            image_norm = image
        overlay = (1 - self.alpha) * image_norm + self.alpha * heatmap_color
        return np.clip(overlay, 0, 1)

    def plot_single(self, image, heatmap, prediction, save_path=None):
        overlay = self.overlay_heatmap(image, heatmap)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image.astype(np.uint8) if image.max() > 1 else image)
        axes[0].set_title("Original Image")
        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay ({prediction})")
        for ax in axes:
            ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_comparison(self, image, heatmaps, titles, save_path=None):
        n = len(heatmaps) + 1
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        axes[0].imshow(image.astype(np.uint8) if image.max() > 1 else image)
        axes[0].set_title("Original")
        for i, (hm, title) in enumerate(zip(heatmaps, titles)):
            overlay = self.overlay_heatmap(image, hm)
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(title)
        for ax in axes:
            ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_top_classes(self, image, heatmaps_per_class, class_names, save_path=None):
        n = len(class_names)
        fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
        for i, (cls_name, hm) in enumerate(zip(class_names, heatmaps_per_class)):
            axes[0, i].imshow(hm, cmap="jet")
            axes[0, i].set_title(f"Heatmap: {cls_name}")
            overlay = self.overlay_heatmap(image, hm)
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f"Overlay: {cls_name}")
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
