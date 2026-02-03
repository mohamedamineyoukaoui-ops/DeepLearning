"""
Football Analytics API - FastAPI Backend with GPU Acceleration
Optimized for real-time video processing with CUDA
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import skimage.color
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import uvicorn

# ============= INITIALIZATION =============

app = FastAPI(
    title="Football Analytics API",
    description="Real-time football player detection with GPU acceleration",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = Path(__file__).parent / "static"
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# ============= GPU OPTIMIZATION =============

def setup_gpu():
    """Configure GPU for optimal performance"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    print("⚠️ No GPU available, using CPU")
    return False

# ============= MODEL LOADING =============

class FootballDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_players = None
        self.model_keypoints = None
        self.is_ready = False
        
        # Detection settings
        self.player_conf = 0.6
        self.keypoint_conf = 0.7
        self.inference_size = 480
        self.keypoint_displacement_tol = 7
        self.num_palette_colors = 3
        
        # Team colors (default)
        self.colors_dic = {
            "Team A": [(41, 71, 138), (220, 98, 88)],
            "Team B": [(144, 200, 255), (188, 199, 3)]
        }
        self.team_colors_lab = None
        self._update_lab_colors()
        
        # Homography cache
        self.homography = None
        self.keypoint_interval = 5
        self.frame_count = 0
        
        # Load config files
        self._load_configs()
        
    def _load_configs(self):
        """Load keypoint positions and label mappings"""
        try:
            with open(BASE_DIR / "pitch map labels position.json", 'r') as f:
                self.keypoints_map_pos = json.load(f)
            
            import yaml
            with open(BASE_DIR / "config pitch dataset.yaml", 'r') as f:
                self.classes_names_dic = yaml.safe_load(f)['names']
            
            with open(BASE_DIR / "config players dataset.yaml", 'r') as f:
                self.labels_dic = yaml.safe_load(f)['names']
                
            # Load tactical map
            tac_map_path = BASE_DIR / "tactical map.jpg"
            self.tac_map = cv2.imread(str(tac_map_path))
            
        except Exception as e:
            print(f"Warning: Could not load config files: {e}")
            self.keypoints_map_pos = {}
            self.classes_names_dic = {}
            self.labels_dic = {0: "player", 1: "referee", 2: "ball"}
            self.tac_map = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    def _update_lab_colors(self):
        """Update LAB color space for team prediction"""
        colors_list = list(self.colors_dic.values())
        flat_colors = [c for team in colors_list for c in team]
        self.team_colors_lab = np.array([
            skimage.color.rgb2lab([i/255 for i in c]) for c in flat_colors
        ])
    
    def load_models(self):
        """Load YOLO models with GPU optimization"""
        print("Loading models...")
        
        try:
            # Players model
            players_path = MODELS_DIR / "Yolo8L Players" / "weights" / "best.pt"
            self.model_players = YOLO(str(players_path))
            self.model_players.to(self.device)
            self.model_players.fuse()
            
            # Keypoints model
            keypoints_path = MODELS_DIR / "Yolo8M Field Keypoints" / "weights" / "best.pt"
            self.model_keypoints = YOLO(str(keypoints_path))
            self.model_keypoints.to(self.device)
            self.model_keypoints.fuse()
            
            # Warmup
            self._warmup()
            
            self.is_ready = True
            print("✅ Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def _warmup(self):
        """Warmup GPU with dummy inference"""
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.model_players(dummy, imgsz=self.inference_size, half=True, verbose=False)
            self.model_keypoints(dummy, imgsz=self.inference_size, half=True, verbose=False)
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("🔥 GPU warmed up!")
    
    def set_team_colors(self, team1_name: str, team1_colors: list, 
                        team2_name: str, team2_colors: list):
        """Update team colors"""
        self.colors_dic = {
            team1_name: [tuple(c) for c in team1_colors],
            team2_name: [tuple(c) for c in team2_colors]
        }
        self._update_lab_colors()
    
    def get_dominant_colors(self, img_rgb, n_colors=5):
        """Fast color extraction using k-means"""
        if img_rgb.size == 0 or img_rgb.shape[0] < 2 or img_rgb.shape[1] < 2:
            return [[128, 128, 128]] * n_colors
        
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        if len(pixels) < n_colors:
            return [list(pixels[0])] * n_colors if len(pixels) > 0 else [[128, 128, 128]] * n_colors
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        _, labels, centers = cv2.kmeans(pixels, min(n_colors, len(pixels)), None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        
        unique, counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(-counts)
        
        colors = [list(centers[i].astype(int)) for i in sorted_idx[:n_colors]]
        while len(colors) < n_colors:
            colors.append(colors[-1] if colors else [128, 128, 128])
        return colors
    
    def predict_team(self, palette):
        """Predict team from color palette"""
        if not palette:
            return 0
        
        palette_rgb = np.array(palette) / 255.0
        palette_lab = skimage.color.rgb2lab(palette_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        
        distances = np.linalg.norm(
            palette_lab[:, np.newaxis, :] - self.team_colors_lab[np.newaxis, :, :], 
            axis=2
        )
        
        nbr_team_colors = len(list(self.colors_dic.values())[0])
        min_team_idx = np.argmin(distances, axis=1) // nbr_team_colors
        return int(np.bincount(min_team_idx).argmax())
    
    def transform_points(self, points):
        """Transform points using homography"""
        if len(points) == 0 or self.homography is None:
            return np.array([])
        
        pts = np.array(points)
        ones = np.ones((len(pts), 1))
        pts_h = np.hstack([pts, ones])
        
        transformed = pts_h @ self.homography.T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        return transformed
    
    def process_frame(self, frame):
        """Process a single frame and return detections"""
        if not self.is_ready:
            return None
        
        self.frame_count += 1
        start_time = time.time()
        
        # Player detection
        results_players = self.model_players(
            frame, conf=self.player_conf, half=True, verbose=False,
            imgsz=self.inference_size, device=self.device
        )
        
        # Keypoint detection (every N frames)
        if self.frame_count % self.keypoint_interval == 1 or self.homography is None:
            results_keypoints = self.model_keypoints(
                frame, conf=self.keypoint_conf, half=True, verbose=False,
                imgsz=self.inference_size, device=self.device
            )
            
            # Update homography
            boxes_k = results_keypoints[0].boxes
            if len(boxes_k) > 3:
                bboxes_k_c = boxes_k.xywh.cpu().numpy()
                labels_k = boxes_k.cls.cpu().numpy().astype(int)
                
                detected_labels = [self.classes_names_dic.get(i, f"k{i}") for i in labels_k]
                src_pts = bboxes_k_c[:, :2].astype(int)
                
                dst_pts = []
                valid_indices = []
                for i, label in enumerate(detected_labels):
                    if label in self.keypoints_map_pos:
                        dst_pts.append(self.keypoints_map_pos[label])
                        valid_indices.append(i)
                
                if len(dst_pts) > 3:
                    src_pts_valid = src_pts[valid_indices]
                    dst_pts = np.array(dst_pts)
                    self.homography, _ = cv2.findHomography(src_pts_valid, dst_pts)
        
        # Extract detections
        boxes = results_players[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        bboxes_c = boxes.xywh.cpu().numpy()
        labels = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        
        # Process players
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        players = []
        ball = None
        referees = []
        team_keys = list(self.colors_dic.keys())
        
        player_positions = []
        player_teams = []
        
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i].astype(int)
            conf = float(confs[i])
            label = int(labels[i])
            
            if label == 0:  # Player
                # Extract jersey color
                obj_img = frame_rgb[y1:y2, x1:x2]
                if obj_img.size > 0:
                    h, w = obj_img.shape[:2]
                    cx1, cx2 = max(w//2 - w//5, 0), w//2 + w//5
                    cy1, cy2 = max(h//3 - h//5, 0), h//3 + h//5
                    center = obj_img[cy1:cy2, cx1:cx2]
                    
                    palette = self.get_dominant_colors(center, 5)
                    team_idx = self.predict_team(palette)
                    team_name = team_keys[team_idx]
                    team_color = self.colors_dic[team_name][0]
                else:
                    team_idx = 0
                    team_name = team_keys[0]
                    team_color = self.colors_dic[team_name][0]
                
                # Foot position for tactical map
                foot_pos = [bboxes_c[i, 0], bboxes_c[i, 1] + bboxes_c[i, 3] / 2]
                player_positions.append(foot_pos)
                player_teams.append(team_idx)
                
                players.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "team": team_name,
                    "team_color": list(team_color)
                })
                
            elif label == 1:  # Referee
                referees.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf
                })
                
            elif label == 2:  # Ball
                ball = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [int((x1+x2)/2), int((y1+y2)/2)],
                    "confidence": conf
                }
        
        # Transform to tactical map coordinates
        tactical_positions = []
        if len(player_positions) > 0 and self.homography is not None:
            transformed = self.transform_points(player_positions)
            for i, pos in enumerate(transformed):
                tactical_positions.append({
                    "x": int(pos[0]),
                    "y": int(pos[1]),
                    "team": team_keys[player_teams[i]],
                    "team_color": list(self.colors_dic[team_keys[player_teams[i]]][0])
                })
        
        # Ball tactical position
        ball_tactical = None
        if ball and self.homography is not None:
            ball_center = [bboxes_c[labels == 2][0, 0], bboxes_c[labels == 2][0, 1]]
            ball_transformed = self.transform_points([ball_center])
            if len(ball_transformed) > 0:
                ball_tactical = {
                    "x": int(ball_transformed[0, 0]),
                    "y": int(ball_transformed[0, 1])
                }
        
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        return {
            "players": players,
            "referees": referees,
            "ball": ball,
            "ball_tactical": ball_tactical,
            "tactical_positions": tactical_positions,
            "fps": round(fps, 1),
            "processing_time_ms": round(processing_time * 1000, 1),
            "frame_number": self.frame_count
        }


# Global detector instance
detector = FootballDetector()


# ============= API ROUTES =============

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    setup_gpu()
    detector.load_models()


@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def get_status():
    """Get API status"""
    return {
        "ready": detector.is_ready,
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "teams": list(detector.colors_dic.keys())
    }


@app.post("/api/set-teams")
async def set_teams(data: dict):
    """Set team names and colors"""
    try:
        detector.set_team_colors(
            data["team1_name"], data["team1_colors"],
            data["team2_name"], data["team2_colors"]
        )
        return {"success": True, "teams": list(detector.colors_dic.keys())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/set-hyperparams")
async def set_hyperparams(data: dict):
    """Set model hyperparameters"""
    try:
        if "player_conf" in data:
            detector.player_conf = float(data["player_conf"])
        if "keypoint_conf" in data:
            detector.keypoint_conf = float(data["keypoint_conf"])
        if "keypoint_tol" in data:
            detector.keypoint_displacement_tol = int(data["keypoint_tol"])
        if "num_palette_colors" in data:
            detector.num_palette_colors = int(data["num_palette_colors"])
        
        return {
            "success": True,
            "player_conf": detector.player_conf,
            "keypoint_conf": detector.keypoint_conf
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file"""
    try:
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Get video info
        cap = cv2.VideoCapture(str(file_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "success": True,
            "filename": file.filename,
            "frame_count": frame_count,
            "fps": fps,
            "resolution": f"{width}x{height}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/get-frame-players")
async def get_frame_players(data: dict):
    """Get players detected in a specific frame for color picking"""
    try:
        filename = data.get("filename")
        frame_number = data.get("frame_number", 1)
        
        video_path = UPLOADS_DIR / filename
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = cap.read()
        
        if not success:
            cap.release()
            raise HTTPException(status_code=400, detail="Could not read frame")
        
        # Detect players
        results = detector.model_players(frame, conf=0.7, half=True, verbose=False,
                                          imgsz=detector.inference_size, device=detector.device)
        
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract player crops
        player_crops = []
        for i, label in enumerate(labels):
            if int(label) == 0:  # Player
                bbox = bboxes[i].astype(int)
                x1, y1, x2, y2 = bbox
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (60, 80))
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
                    crop_b64 = base64.b64encode(buffer).decode('utf-8')
                    player_crops.append({
                        "image": crop_b64,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        # Full frame for display
        frame_display = cv2.resize(frame, (640, 360))
        _, frame_buffer = cv2.imencode('.jpg', frame_display)
        frame_b64 = base64.b64encode(frame_buffer).decode('utf-8')
        
        # Composite image of all players (2 rows)
        if player_crops:
            n_players = len(player_crops)
            cols = max(n_players // 2 + n_players % 2, 1)
            composite = np.ones((160, cols * 60, 3), dtype=np.uint8) * 255
            
            for i, crop in enumerate(player_crops):
                img_data = base64.b64decode(crop["image"])
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                row = 0 if i < cols else 1
                col = i if i < cols else i - cols
                if col < cols:
                    composite[row*80:(row+1)*80, col*60:(col+1)*60] = img
            
            _, comp_buffer = cv2.imencode('.jpg', composite)
            composite_b64 = base64.b64encode(comp_buffer).decode('utf-8')
        else:
            composite_b64 = None
        
        cap.release()
        
        return {
            "success": True,
            "frame": frame_b64,
            "composite": composite_b64,
            "composite_width": cols * 60 if player_crops else 0,
            "composite_height": 160,
            "player_count": len(player_crops)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pick-color")
async def pick_color(data: dict):
    """Get color at specific pixel from composite image"""
    try:
        filename = data.get("filename")
        frame_number = data.get("frame_number", 1)
        x = data.get("x", 0)
        y = data.get("y", 0)
        
        video_path = UPLOADS_DIR / filename
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            return {"success": False, "error": "Could not read frame"}
        
        # Detect players and build composite
        results = detector.model_players(frame, conf=0.7, half=True, verbose=False,
                                          imgsz=detector.inference_size, device=detector.device)
        
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        player_crops = []
        for i, label in enumerate(labels):
            if int(label) == 0:
                bbox = bboxes[i].astype(int)
                x1, y1, x2, y2 = bbox
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (60, 80))
                    player_crops.append(crop_resized)
        
        if not player_crops:
            return {"success": False, "error": "No players found"}
        
        n_players = len(player_crops)
        cols = max(n_players // 2 + n_players % 2, 1)
        composite = np.ones((160, cols * 60, 3), dtype=np.uint8) * 255
        
        for i, crop in enumerate(player_crops):
            row = 0 if i < cols else 1
            col = i if i < cols else i - cols
            if col < cols:
                composite[row*80:(row+1)*80, col*60:(col+1)*60] = crop
        
        # Get pixel color
        if 0 <= y < composite.shape[0] and 0 <= x < composite.shape[1]:
            color = composite[y, x]
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
            return {
                "success": True,
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "hex": hex_color
            }
        
        return {"success": False, "error": "Coordinates out of bounds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws/video-stream")
async def video_stream(websocket: WebSocket):
    """WebSocket for real-time video processing"""
    await websocket.accept()
    
    try:
        # Receive video filename
        data = await websocket.receive_json()
        video_path = UPLOADS_DIR / data.get("filename", "")
        
        if not video_path.exists():
            await websocket.send_json({"error": "Video not found"})
            return
        
        cap = cv2.VideoCapture(str(video_path))
        detector.frame_count = 0
        detector.homography = None
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame
            result = detector.process_frame(frame)
            
            if result:
                # Encode frame as JPEG for display
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                result["frame"] = frame_base64
                await websocket.send_json(result)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.001)
        
        cap.release()
        await websocket.send_json({"complete": True})
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})


@app.get("/api/tactical-map")
async def get_tactical_map():
    """Get tactical map as base64"""
    if detector.tac_map is not None:
        _, buffer = cv2.imencode('.jpg', detector.tac_map)
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    return {"image": None}


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============= RUN SERVER =============

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
