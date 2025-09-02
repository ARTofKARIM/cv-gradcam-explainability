# Grad-CAM Explainability Toolkit

Visual explanation toolkit for CNN predictions using Grad-CAM and Grad-CAM++ with support for multiple pre-trained architectures (VGG16, ResNet50, MobileNetV2).

## Architecture
```
cv-gradcam-explainability/
├── src/
│   ├── model_loader.py    # Pre-trained model loading (VGG16, ResNet50, etc.)
│   ├── gradcam.py         # Grad-CAM and Grad-CAM++ implementations
│   ├── image_utils.py     # Image loading and preprocessing
│   └── visualization.py   # Heatmap overlay, comparison, multi-class plots
├── config/config.yaml
├── tests/test_gradcam.py
└── main.py
```

## Methods
| Method | Description |
|--------|-------------|
| Grad-CAM | Gradient-weighted Class Activation Mapping |
| Grad-CAM++ | Improved weighting for better localization |

## Installation
```bash
git clone https://github.com/mouachiqab/cv-gradcam-explainability.git
cd cv-gradcam-explainability
pip install -r requirements.txt
```

## Usage
```bash
python main.py --image data/cat.jpg --model VGG16 --method both
```

## Technologies
- Python 3.9+, TensorFlow/Keras, OpenCV, matplotlib






