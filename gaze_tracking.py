import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from datetime import datetime

class GazeTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.pose = mp.solutions.pose.Pose()
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        self.eye_indices = {
            'left': [362, 385, 387, 263, 373, 380],
            'right': [33, 160, 158, 133, 153, 144]
        }
        self.data = []
        self.start_time = datetime.now()
        self.output_file = "gaze_tracking_data.xlsx"
        self.last_logged_time = 0  # Store last recorded timestamp in seconds

    def get_eye_aspect_ratio(self, eye_points):
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        return (A + B) / (2.0 * C)

    def get_pupil_position(self, eye_region):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        _, threshold_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    def get_head_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks:
            head_x = results.pose_landmarks.landmark[0].x
            head_y = results.pose_landmarks.landmark[0].y
            head_z = results.pose_landmarks.landmark[0].z
            return head_x, head_y, head_z
        return None

    def process_frame(self, frame, timestamp):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        left_eye_status, right_eye_status = "", ""
        left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y = None, None, None, None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for eye in ['left', 'right']:
                    eye_points = [(int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0])) for i in self.eye_indices[eye]]
                    for point in eye_points:
                        cv2.circle(frame, point, 2, (0, 255, 0), -1)
                    ear = self.get_eye_aspect_ratio(eye_points)
                    blink_detected = ear < 0.2
                    x_min, x_max = min(p[0] for p in eye_points), max(p[0] for p in eye_points)
                    y_min, y_max = min(p[1] for p in eye_points), max(p[1] for p in eye_points)
                    eye_region = frame[y_min:y_max, x_min:x_max]
                    pupil_position = self.get_pupil_position(eye_region)
                    if eye == 'left':
                        left_eye_status = "Blinking" if blink_detected else "Not Blinking"
                        if pupil_position:
                            left_pupil_x, left_pupil_y = pupil_position
                    else:
                        right_eye_status = "Blinking" if blink_detected else "Not Blinking"
                        if pupil_position:
                            right_pupil_x, right_pupil_y = pupil_position
                    cv2.putText(frame, f"{eye.capitalize()} Eye: {'Blinking' if blink_detected else 'Not Blinking'}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.drawing_utils.draw_landmarks(frame, face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION, self.drawing_spec)
        head_pose = self.get_head_pose(frame)
        head_x, head_y, head_z = head_pose if head_pose else (None, None, None)
        cv2.putText(frame, f"Head Pose: {head_pose}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if timestamp - self.last_logged_time >= 60:
            print(f"Timestamp: {timestamp}, Left Eye: {left_eye_status}, Right Eye: {right_eye_status}, Left Pupil: ({left_pupil_x}, {left_pupil_y}), Right Pupil: ({right_pupil_x}, {right_pupil_y}), Head Pose: ({head_x}, {head_y}, {head_z})")
            self.data.append([timestamp, left_eye_status, right_eye_status, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y, head_x, head_y, head_z])
            self.last_logged_time = timestamp
        return frame

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = (datetime.now() - self.start_time).total_seconds()
            frame = self.process_frame(frame, timestamp)
            cv2.imshow('Gaze Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        df = pd.DataFrame(self.data, columns=["Timestamp", "Left Eye Status", "Right Eye Status", "Left Pupil X", "Left Pupil Y", "Right Pupil X", "Right Pupil Y", "Head X", "Head Y", "Head Z"])
        df.to_excel(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")

# Usage
gaze_tracker = GazeTracker("video.mp4")
gaze_tracker.run()
