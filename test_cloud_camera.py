import cv2
import requests

# Your live deployed Render API endpoint
CLOUD_URL = "https://medical-waste-cloud-api.onrender.com/predict"

cap = cv2.VideoCapture(0)

print("==================================================")
print("   SMART MEDICAL WASTE CLOUD API TESTER           ")
print("==================================================")
print("1. Point camera at an item.")
print("2. Press 's' to send image to Render Cloud AI.")
print("3. Press 'q' to quit.")
print("==================================================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not access camera.")
        break

    cv2.imshow("Smart Medical Waste - Press 's' for Cloud AI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("Sending frame to Render Cloud Server...")

        # Encode current frame to JPEG format
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

        try:
            # Send HTTP POST request to your cloud API
            response = requests.post(CLOUD_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("\n" + "=" * 45)
                print("         LIVE CLOUD INFERENCE RESULT          ")
                print("=" * 45)
                print(f"Is Medical Waste : {data.get('is_medical_waste')}")
                print(f"Predicted Class  : {data.get('class')}")
                print(f"Confidence       : {data.get('confidence_percentage')}%")
                print(f"Target Bin       : {data.get('target_bin')}")
                print("=" * 45 + "\n")
            else:
                print(f"Server Returned Error Code: {response.status_code}")

        except Exception as e:
            print(f"Failed to communicate with Cloud API: {e}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()