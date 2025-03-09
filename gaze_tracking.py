import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from datetime import datetime

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.eye_indices = {
            'left': [362, 385, 387, 263, 373, 380],
            'right': [33, 160, 158, 133, 153, 144]
        }
        self.left_eye_features = None
        self.right_eye_features = None

    def get_eye_aspect_ratio(self, eye_points):
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    def get_pupil_position(self, eye_region):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
        return None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.left_eye_features = self.extract_eye_features(face_landmarks, 'left', frame)
                self.right_eye_features = self.extract_eye_features(face_landmarks, 'right', frame)
        return frame

    def extract_eye_features(self, face_landmarks, eye, frame):
        eye_points = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                       int(face_landmarks.landmark[i].y * frame.shape[0]))
                      for i in self.eye_indices[eye]]
        ear = self.get_eye_aspect_ratio(eye_points)
        blink_detected = ear < 0.2

        x_min = min(p[0] for p in eye_points)
        x_max = max(p[0] for p in eye_points)
        y_min = min(p[1] for p in eye_points)
        y_max = max(p[1] for p in eye_points)
        eye_region = frame[y_min:y_max, x_min:x_max]

        pupil_position = self.get_pupil_position(eye_region)
        if pupil_position:
            pupil_position = (pupil_position[0] + x_min, pupil_position[1] + y_min)

        return {
            'ear': ear,
            'blink_detected': blink_detected,
            'pupil_position': pupil_position
        }

def initialize_excel_file(filename='eye_features.xlsx'):
    columns = ['Timestamp', 'Left Eye EAR', 'Right Eye EAR',
               'Left Pupil Position', 'Right Pupil Position',
               'Left Eye Blink Status', 'Right Eye Blink Status']
    df = pd.DataFrame(columns=columns)
    df.to_excel(filename, index=False)

def log_eye_features(gaze_tracker, filename='eye_features.xlsx'):
    if gaze_tracker.left_eye_features and gaze_tracker.right_eye_features:
        data = {
            'Timestamp': datetime.now(),
            'Left Eye EAR': gaze_tracker.left_eye_features['ear'],
            'Right Eye EAR': gaze_tracker.right_eye_features['ear'],
            'Left Pupil Position': gaze_tracker.left_eye_features['pupil_position'],
            'Right Pupil Position': gaze_tracker.right_eye_features['pupil_position'],
            'Left Eye Blink Status': 'Blinking' if gaze_tracker.left_eye_features['blink_detected'] else 'Not Blinking',
            'Right Eye Blink Status': 'Blinking' if gaze_tracker.right_eye_features['blink_detected'] else 'Not Blinking'
        }
        df = pd.DataFrame([data])
        with pd.ExcelWriter(filename, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

# Initialize the Excel file
initialize_excel_file()

# Create the GazeTracker instance
gaze_tracker = GazeTracker()

# Path to your video file
video_path = 'path_to_your_video.mp4'

# Start video capture from the file
cap = cv2.VideoCapture(video_path)

# Retrieve the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(1000 / fps)  # Time between frames in milliseconds

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = gaze_tracker.process_frame(frame)
        cv2.imshow('Gaze Tracker', frame)

        # Log eye features for each frame
        log_eye_features(gaze_tracker)

        if cv2.waitKey(frame_interval) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
