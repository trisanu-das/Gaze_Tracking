#imports
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from datetime import datetime

blinks = 0

#The Class

class GazeTracker:
    def __init__(self, video_path, out_path):
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
        self.output_file = out_path
        self.last_logged_time = 0  # Store last logged timestamp in seconds

    def get_eye_aspect_ratio(self, eye_points):
        A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    def get_pupil_position(self, eye_region):
        if eye_region is None or eye_region.size == 0:
            return None  # Skip if empty
        
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        # Adaptive thresholding for better segmentation
        threshold_eye = cv2.adaptiveThreshold(gray_eye, 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
        # Optional: Display the thresholded eye for debugging
        # cv2.imshow("Threshold Eye", threshold_eye)
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
        return None

    def get_head_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks:
            head_x = results.pose_landmarks.landmark[0].x
            head_y = results.pose_landmarks.landmark[0].y
            head_z = results.pose_landmarks.landmark[0].z
            return (head_x, head_y, head_z)
        return (None, None, None)

    def process_frame(self, frame, timestamp):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize feature variables
        left_status, left_pupil_x, left_pupil_y = None, None, None
        right_status, right_pupil_x, right_pupil_y = None, None, None
        head_pose = (None, None, None)

        if results.multi_face_landmarks:
            global blinks
            # Use the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            # ----- LEFT EYE -----
            left_eye_points = [
                (int(face_landmarks.landmark[i].x * frame.shape[1]),
                 int(face_landmarks.landmark[i].y * frame.shape[0]))
                for i in self.eye_indices['left']
            ]
            left_ear = self.get_eye_aspect_ratio(left_eye_points)
            left_status = "Blinking" if left_ear < 0.2 else "Not Blinking"
            blinks += (1 if left_ear < 0.2 else 0)
            x_min_left = min(p[0] for p in left_eye_points)
            x_max_left = max(p[0] for p in left_eye_points)
            y_min_left = min(p[1] for p in left_eye_points)
            y_max_left = max(p[1] for p in left_eye_points)
            if y_max_left > y_min_left and x_max_left > x_min_left:
                left_eye_region = frame[y_min_left:y_max_left, x_min_left:x_max_left]
                left_pupil = self.get_pupil_position(left_eye_region)
                if left_pupil:
                    left_pupil_x, left_pupil_y = left_pupil
                else:
                    left_pupil_x, left_pupil_y = None, None
            else:
                left_pupil_x, left_pupil_y = None, None
            
            # ----- RIGHT EYE -----
            right_eye_points = [
                (int(face_landmarks.landmark[i].x * frame.shape[1]),
                 int(face_landmarks.landmark[i].y * frame.shape[0]))
                for i in self.eye_indices['right']
            ]
            right_ear = self.get_eye_aspect_ratio(right_eye_points)
            right_status = "Blinking" if right_ear < 0.2 else "Not Blinking"
            blinks += (1 if left_ear < 0.2 else 0)
            x_min_right = min(p[0] for p in right_eye_points)
            x_max_right = max(p[0] for p in right_eye_points)
            y_min_right = min(p[1] for p in right_eye_points)
            y_max_right = max(p[1] for p in right_eye_points)
            if y_max_right > y_min_right and x_max_right > x_min_right:
                right_eye_region = frame[y_min_right:y_max_right, x_min_right:x_max_right]
                right_pupil = self.get_pupil_position(right_eye_region)
                if right_pupil:
                    right_pupil_x, right_pupil_y = right_pupil
                else:
                    right_pupil_x, right_pupil_y = None, None
            else:
                right_pupil_x, right_pupil_y = None, None
            
            
            # Get head pose
            head_pose = self.get_head_pose(frame)
            if head_pose and head_pose[0] is not None:
                head_x, head_y, head_z = head_pose
        else:
            # No face detected; skip logging for this frame.
            return frame

        # Log the data every 60 seconds
        if (head_pose[0] is not None and 
            timestamp - self.last_logged_time >= 60):
            row = [timestamp, head_pose[0], head_pose[1], head_pose[2],
                   left_status, left_pupil_x, left_pupil_y,
                   right_status, right_pupil_x, right_pupil_y]
            self.data.append(row)
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        columns = ["Timestamp", "Head Pose X", "Head Pose Y", "Head Pose Z",
                   "Left Eye Status", "Left Pupil X", "Left Pupil Y",
                   "Right Eye Status", "Right Pupil X", "Right Pupil Y"]
        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")
