# Blood Pressure Prediction API

This FastAPI application provides endpoints for predicting blood pressure from both images and videos using facial analysis.

## Features

- Image-based blood pressure prediction
- Video-based blood pressure prediction
- Real-time processing
- CORS enabled for cross-origin requests

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fastapi_backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place model files in the root directory:
- `ridge_model_systolic.pkl`
- `lasso_model_diastolic.pkl`

## Running the Application

1. Start the server:
```bash
uvicorn combined_app:app --host 0.0.0.0 --port 56379
```

2. The API will be available at:
- Local: `http://localhost:56379`
- Render: `https://your-app-name.onrender.com`

## API Endpoints

### 1. Image Prediction
- **Endpoint**: `/predict`
- **Method**: POST
- **Input**: Image file
- **Response**: JSON with systolic and diastolic predictions

### 2. Video Prediction
- **Endpoint**: `/predict_from_video`
- **Method**: POST
- **Input**: Video file
- **Response**: JSON with systolic and diastolic predictions

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn combined_app:app --host 0.0.0.0 --port $PORT`
4. Add environment variables if needed
5. Deploy!

## Environment Variables

- `PORT`: Port number (default: 56379)
- `MODEL_PATH`: Path to model files (default: current directory)

## License

[Your License Here] 