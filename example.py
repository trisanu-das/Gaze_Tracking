import keyboard
from .gaze_tracking import GazeTracker

cap = cv2.VideoCapture(0)
gaze_tracker = GazeTracker()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if keyboard.is_pressed("a"):
        break
    frame = gaze_tracker.process_frame(frame)
    cv2.imshow('Gaze Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
