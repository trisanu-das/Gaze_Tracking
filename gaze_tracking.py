import cv2
import numpy as np
import mediapipe as mp

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
        self.eye_indices = {
            'left': [362, 385, 387, 263, 373, 380],
            'right': [33, 160, 158, 133, 153, 144]
        }

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
                for eye in ['left', 'right']:
                    eye_points = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                   int(face_landmarks.landmark[i].y * frame.shape[0]))
                                  for i in self.eye_indices[eye]]
                    
                    # Draw eye landmarks
                    for point in eye_points:
                        cv2.circle(frame, point, 2, (0, 255, 0), -1)

                    ear = self.get_eye_aspect_ratio(eye_points)
                    blink_detected = ear < 0.2

                    # Extract eye region for pupil detection
                    x_min = min(p[0] for p in eye_points)
                    x_max = max(p[0] for p in eye_points)
                    y_min = min(p[1] for p in eye_points)
                    y_max = max(p[1] for p in eye_points)
                    eye_region = frame[y_min:y_max, x_min:x_max]
                    
                    pupil_position = self.get_pupil_position(eye_region)
                    if pupil_position:
                        pupil_x = pupil_position[0] + x_min
                        pupil_y = pupil_position[1] + y_min
                        cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 0, 255), -1)

                    # Display blink status
                    status = "Blinking" if blink_detected else "Not Blinking"
                    cv2.putText(frame, f"{eye.capitalize()} Eye: {status}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame



# Also shows the position of the eye in the actual frame
