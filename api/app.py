# app.py
import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask App and MediaPipe Pose
app = Flask(__name__)
mp_pose = mp.solutions.pose

# --- Helper Function to Calculate Angles ---
# This is the same logic from the original script
def calculate_angle(a, b, c):
    """Calculates the angle between three 2D points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- API Endpoint to Process Videos ---
@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    # Check if a video file was uploaded
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']

    # We need a temporary path to save the uploaded file
    # since OpenCV works with file paths, not in-memory files
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    # --- Core Push-up Counting Logic ---
    cap = cv2.VideoCapture(video_path)
    counter = 0
    stage = 'up'

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            image.flags.writeable = True

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for the left arm
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate elbow angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Push-up counter logic
                if angle > 160 and stage == 'down':
                    stage = "up"
                    counter += 1
                if angle < 90:
                    stage = "down"
            except Exception as e:
                # If landmarks are not detected, just continue
                pass

    cap.release()
    # Clean up the temporary file
    os.remove(video_path)

    # Return the final count as a JSON response
    return jsonify({'pushup_count': counter})

if __name__ == '__main__':
    # This part is for local development. Gunicorn will run the app in production.
    # The host='0.0.0.0' makes it accessible on your local network.
    app.run(debug=True, host='0.0.0.0')