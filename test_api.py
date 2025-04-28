import requests

url = "http://127.0.0.1:56379/predict_from_video"
# url = "http://localhost:56379/predict"
try:
    # Try to make a request without a file to see if the server responds
    response = requests.post(url)
    print("API Response Status:", response.status_code)
    print("API Response:", response.text)
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the API server. Make sure it's running!")
except Exception as e:
    print(f"Error: {str(e)}") 