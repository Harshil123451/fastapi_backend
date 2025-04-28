import requests

url = "http://127.0.0.1:56379/predict_from_video"
video_path = "test_video.mp4"  # Replace with your video path

try:
    with open(video_path, 'rb') as video:
        files = {'file': video}
        response = requests.post(url, files=files)
        print("Response:", response.json())
except FileNotFoundError:
    print(f"Error: Could not find video file at {video_path}")
except Exception as e:
    print(f"Error: {str(e)}") 