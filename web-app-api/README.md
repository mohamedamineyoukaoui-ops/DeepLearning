# Football Analytics Web App API

A high-performance REST API for real-time football player detection with GPU acceleration.

## 🚀 Features

- **GPU Accelerated**: Uses CUDA for fast inference (~30+ FPS)
- **Real-time WebSocket**: Live video processing with instant feedback
- **Modern UI**: Sleek vanilla JavaScript frontend
- **Team Detection**: Automatic team classification by jersey color
- **Tactical Map**: Real-time player position mapping

## 📁 Structure

```
web-app-api/
├── api.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Frontend (vanilla JS)
└── uploads/            # Uploaded videos (auto-created)
```

## 🔧 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python api.py
```

Or with uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

3. Open browser at: **http://localhost:8000**

## 🎮 Usage

1. **Upload Video**: Drag & drop or click to upload a football video
2. **Configure Teams**: Set team names and jersey colors
3. **Start Detection**: Click "Start Detection" to begin processing
4. **View Results**: Watch real-time detections on video and tactical map

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend UI |
| GET | `/api/status` | Check API and GPU status |
| POST | `/api/set-teams` | Configure team names/colors |
| POST | `/api/upload-video` | Upload video file |
| WS | `/ws/video-stream` | WebSocket for real-time processing |
| GET | `/api/tactical-map` | Get tactical map image |

## ⚡ Performance Tips

- Uses FP16 (half precision) for faster GPU inference
- Keypoint detection runs every 5 frames
- Image size optimized to 480px for speed
- GPU warmup eliminates first-frame latency

## 📊 Expected Performance

| GPU | Expected FPS |
|-----|-------------|
| RTX 4050 | 25-35 FPS |
| RTX 3060 | 30-40 FPS |
| RTX 4080 | 50-60 FPS |

## 🛠️ Configuration

Edit `api.py` to adjust:
- `inference_size`: Image size (default: 480)
- `player_conf`: Player detection threshold (default: 0.6)
- `keypoint_conf`: Keypoint detection threshold (default: 0.7)
- `keypoint_interval`: Keypoint detection frequency (default: 5)
