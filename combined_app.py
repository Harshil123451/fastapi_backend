from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
import joblib
import tempfile
import os
import warnings
import uvicorn
from PIL import Image
import io
from typing import Dict, Any, Union

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = FastAPI(
    title="Blood Pressure Prediction API",
    description="API for predicting blood pressure from facial images and videos",
    version="1.0.0"
)

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ridge_model = joblib.load("ridge_model_systolic.pkl")
        lasso_model = joblib.load("lasso_model_diastolic.pkl")
except FileNotFoundError:
    print("Warning: Model files not found. Models will be loaded when available.")
    ridge_model = None
    lasso_model = None

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """Root endpoint that returns API information"""
    return {
        "message": "Welcome to Blood Pressure Prediction API",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "image_prediction": "/predict",
            "video_prediction": "/predict_from_video"
        }
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": str(ridge_model is not None and lasso_model is not None)
    }

def bandpass_filter(signal, lowcut=0.75, highcut=3.5, fs=30, order=5):
    if len(signal) < 34:  # If signal is too short, pad it
        pad_length = 34 - len(signal)
        signal = np.pad(signal, (0, pad_length), mode='edge')
    
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def estimate_hr_from_signal(signal, fps=30):
    if len(signal) < fps:
        signal = np.pad(signal, (0, fps - len(signal)), mode='edge')
    
    signal = np.array(signal) - np.mean(signal)
    filtered = bandpass_filter(signal, fs=fps)
    fft_vals = np.abs(rfft(filtered))
    freqs = rfftfreq(len(filtered), 1 / fps)
    valid = (freqs >= 0.75) & (freqs <= 3.5)
    
    if not np.any(valid):
        return 75  # Return default HR if no valid frequency found
    dominant_freq = freqs[valid][np.argmax(fft_vals[valid])]
    return dominant_freq * 60

def extract_forehead_mean(frame, face_mesh):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        points = [lm[i] for i in [10, 338, 297]]  # Forehead landmarks
        coords = [(int(p.x * w), int(p.y * h)) for p in points]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(coords, np.int32), 255)

        green = np.mean(frame[:, :, 1][mask == 255])
        red = np.mean(frame[:, :, 2][mask == 255])
        return green, red
    return None, None

def process_image(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    green_values = []
    red_values = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        green, red = extract_forehead_mean(frame, face_mesh)
        if green is not None and red is not None:
            green_values = [green] * 30  # Simulate 30 frames
            red_values = [red] * 30      # Simulate 30 frames
        else:
            raise ValueError("No face detected in the image")

    # Calculate features
    green_mean = np.mean(green_values)
    green_std = np.std(green_values)
    red_mean = np.mean(red_values)
    red_std = np.std(red_values)
    est_hr = estimate_hr_from_signal(green_values)

    best_mean = green_mean if green_std < red_std else red_mean
    best_std = green_std if green_std < red_std else red_std

    return [best_mean, best_std, est_hr, est_hr]

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default to 30 fps if not detected
    
    green_values = []
    red_values = []
    frames_to_process = 45 * fps  # 45 seconds worth of frames
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        frame_count = 0
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
                
            green, red = extract_forehead_mean(frame, face_mesh)
            if green is not None and red is not None:
                green_values.append(green)
                red_values.append(red)
            frame_count += 1

    cap.release()
    
    if len(green_values) < fps:  # If we don't have enough frames
        raise HTTPException(status_code=400, detail="Not enough valid frames with face detection")

    # Calculate features
    green_mean = np.mean(green_values)
    green_std = np.std(green_values)
    red_mean = np.mean(red_values)
    red_std = np.std(red_values)
    est_hr = estimate_hr_from_signal(green_values, fps)

    best_mean = green_mean if green_std < red_std else red_mean
    best_std = green_std if green_std < red_std else red_std

    return [best_mean, best_std, est_hr, est_hr]

def make_predictions(features):
    if ridge_model is None or lasso_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    systolic_pred = int(ridge_model.predict([features])[0])
    diastolic_pred = int(lasso_model.predict([features])[0])
    
    # Ensure predictions are within reasonable ranges
    systolic_pred = max(90, min(systolic_pred, 180))
    diastolic_pred = max(60, min(diastolic_pred, 120))
    
    return {
        "systolic": systolic_pred,
        "diastolic": diastolic_pred
    }

@app.post("/predict")
async def predict_bp(file: UploadFile = File(...)) -> Dict[str, int]:
    """
    Predict blood pressure from an image file.
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Dictionary containing systolic and diastolic predictions
    """
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Process image and extract features
        features = process_image(image_bytes)
        
        # Make predictions
        return make_predictions(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_from_video")
async def predict_bp_from_video(file: UploadFile = File(...)) -> Dict[str, int]:
    """
    Predict blood pressure from a video file.
    
    Args:
        file: Video file (MP4)
        
    Returns:
        Dictionary containing systolic and diastolic predictions
    """
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await file.read()
            temp_video.write(content)
            temp_video_path = temp_video.name

        try:
            # Process video and extract features
            features = process_video(temp_video_path)
            
            # Make predictions
            return make_predictions(features)
        finally:
            # Clean up temporary file
            os.unlink(temp_video_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=56379) 