"""Main entry point for Grad-CAM explainability toolkit."""
import argparse
import yaml
from src.model_loader import ModelLoader
from src.gradcam import GradCAM, GradCAMPlusPlus
from src.image_utils import ImageUtils
from src.visualization import GradCAMVisualizer

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Explainability")
    parser.add_argument("--image", required=True, help="Image path")
    parser.add_argument("--model", default="VGG16", choices=["VGG16", "ResNet50", "MobileNetV2"])
    parser.add_argument("--method", default="gradcam", choices=["gradcam", "gradcam++", "both"])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = ModelLoader(args.model)
    orig_img, resized_img = ImageUtils.load_image(args.image, tuple(config["model"]["input_size"]))
    input_tensor = ImageUtils.prepare_input(resized_img, loader.preprocess_fn)
    preds, top5 = loader.predict(resized_img)
    print(f"Top prediction: {top5[0][1]} ({top5[0][2]:.2%})")

    viz = GradCAMVisualizer(config["output"]["dir"], config["gradcam"]["alpha"])
    layer = config["model"].get("layer_name") or loader.get_last_conv_layer()

    if args.method in ["gradcam", "both"]:
        gc = GradCAM(loader.model, layer)
        heatmap, cls = gc.compute(input_tensor)
        viz.plot_single(resized_img, heatmap, top5[0][1], f"{config['output']['dir']}gradcam_{args.model}.png")
    if args.method in ["gradcam++", "both"]:
        gcpp = GradCAMPlusPlus(loader.model, layer)
        heatmap, cls = gcpp.compute(input_tensor)
        viz.plot_single(resized_img, heatmap, top5[0][1], f"{config['output']['dir']}gradcampp_{args.model}.png")
    print("Done.")

if __name__ == "__main__":
    main()
