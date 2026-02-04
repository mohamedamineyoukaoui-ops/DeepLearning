# ⚽ Tactical Vision: Football Analytics with Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Transform raw football video into tactical intelligence using Computer Vision and Deep Learning**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Performance](#-performance)

</div>

---

## 📖 Overview

**Tactical Vision** is a football analysis system that detects players, referees, and the ball from tactical camera footage, predicts team assignments based on jersey colors, and projects positions onto a 2D tactical map in real-time.

### 🎯 Project Goals
- Automate football match analysis
- Provide real-time tactical visualization
- Enable data-driven decision making for coaches and analysts

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏃 **Player Detection** | Detect players, referees, and ball using YOLOv8 |
| 👕 **Team Prediction** | Classify players by team using jersey color analysis (K-means + LAB color space) |
| 🗺️ **Tactical Map** | Project player positions onto a 2D pitch representation via homography |
| ⚽ **Ball Tracking** | Track ball movement across frames with position history |
| 📍 **Field Keypoints** | Detect 28 field landmarks for accurate coordinate transformation |
| 🚀 **GPU Accelerated** | FP16 inference with CUDA optimization for ~40 FPS processing |

---

## 🎬 Demo

### Streamlit Web Application
![Application Workflow](workflow%20diagram.png)

### Sample Output
The system provides side-by-side visualization:
- **Left**: Annotated video with player detection and team colors
- **Right**: Real-time tactical map with player positions

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Video   │────▶│  YOLOv8 Models   │────▶│   Homography    │
│  (Tactical Cam) │     │  • Players (L)   │     │  Transformation │
└─────────────────┘     │  • Keypoints (M) │     └────────┬────────┘
                        └──────────────────┘              │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Tactical Map   │◀────│ Team Prediction  │◀────│ Color Analysis  │
│   Projection    │     │   (K-means+LAB)  │     │  (Jersey RGB)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Models

| Model | Architecture | Parameters | Purpose | Training |
|-------|--------------|------------|---------|----------|
| Players | YOLOv8L | 43.7M | Detect players, refs, ball | 30 epochs |
| Keypoints | YOLOv8M | 25.9M | Detect 28 field landmarks | 20 epochs |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- 6GB+ VRAM for optimal performance

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Tactical-Vision-Football-Analytics.git
cd Tactical-Vision-Football-Analytics
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate football-analytics
```

Or install with pip:
```bash
pip install -r requirements.txt
```

3. **Download model weights**

Download the pre-trained weights and place them in the `models/` directory:
- [Players Model (YOLOv8L) - 83MB](https://drive.google.com/your-link) → `models/Yolo8L Players/weights/best.pt`
- [Keypoints Model (YOLOv8M) - 50MB](https://drive.google.com/your-link) → `models/Yolo8M Field Keypoints/weights/best.pt`

4. **Download demo videos**

Place demo videos in the `Streamlit web app/` directory:
- [Demo Video 1](https://drive.google.com/your-link) → `demo_vid_1.mp4`
- [Demo Video 2](https://drive.google.com/your-link) → `demo_vid_2.mp4`

---

## 🚀 Usage

### Streamlit Web Application

```bash
cd "Streamlit web app"
streamlit run main.py
```

Then open `http://localhost:8501` in your browser.

### FastAPI Backend

```bash
cd web-app-api
uvicorn api:app --host 0.0.0.0 --port 8000
```

API available at `http://localhost:8000`

### Jupyter Notebook

For experimentation and analysis:
```bash
jupyter notebook "Football Object Detection With Tactical Map.ipynb"
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Mean FPS | **39.78** |
| Player Inference | 15.13 ms |
| Keypoint Inference | 11.37 ms |
| Mean Players/Frame | 18.5 |
| Mean Confidence | 83.12% |
| Ball Detection Rate | 29.5% |
| Homography Success | 100% |

*Tested on NVIDIA RTX 4050 (6GB VRAM)*

---

## 📁 Project Structure

```
Tactical-Vision/
├── Football Object Detection With Tactical Map.ipynb   # Main notebook
├── evaluation_metrics.csv                              # Performance metrics
├── tactical map.jpg                                    # 2D pitch template
├── config players dataset.yaml                         # Player class config
├── config pitch dataset.yaml                           # Keypoint class config
├── pitch map labels position.json                      # Keypoint coordinates
│
├── models/
│   ├── Yolo8L Players/weights/best.pt                  # Player detection model
│   └── Yolo8M Field Keypoints/weights/best.pt          # Keypoint detection model
│
├── Streamlit web app/
│   ├── main.py                                         # Streamlit UI
│   ├── detection.py                                    # Detection logic
│   └── outputs/                                        # Saved results
│
└── web-app-api/
    ├── api.py                                          # FastAPI backend
    ├── static/index.html                               # Web frontend
    └── uploads/                                        # Uploaded videos
```

---

## 📚 Datasets

The models were trained on custom annotated datasets:

| Dataset | Images | Annotations | Classes |
|---------|--------|-------------|---------|
| Players Dataset | 600 | 12,801 | Player, Referee, Ball |
| Field Keypoints | 342 | 4,747 | 28 field landmarks |

---

## ⚠️ Known Limitations

| Limitation | Impact | Proposed Solution |
|------------|--------|-------------------|
| Intermittent ball detection | High | Lower threshold + temporal interpolation |
| Occluded players | Moderate | Multi-object tracker (DeepSORT/ByteTrack) |
| Team color confusion | Low (~15%) | Fine-tuning color picker |
| HD resolution latency | Variable | TensorRT optimization |

---

## 🔮 Future Work

- [ ] Heat map generation
- [ ] Player distance tracking
- [ ] Formation analysis
- [ ] Pass network visualization
- [ ] Event detection (goals, fouls, etc.)

---

## 👥 Authors

- Mohamed Amine Youkaoui
- Mohamed Ali Echchachoui
- Adam Bakhadi
- Abdelwahab Laksiri
- Sahhar El Mehdi
- Elamchi Omar

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [scikit-image](https://scikit-image.org/)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

</div>
