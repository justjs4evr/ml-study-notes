import numpy as np
import cv2
from ultralytics import YOLO

# Initialize YOLOv8 Pose model
# This will automatically download 'yolov8n-pose.pt' (approx 6MB) on first run
model = YOLO('yolov8n-pose.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

def detect_gesture(keypoints):
    """
    Gesture detection based on Body Pose keypoints (COCO format).
    Note: Standard Pose models track body joints (wrists, elbows), not fingers.
    """
    # keypoints shape: (17, 3) -> [x, y, confidence]
    # COCO Indices: 0:Nose, 9:Left Wrist, 10:Right Wrist
    
    nose_y = keypoints[0][1]
    left_wrist_y = keypoints[9][1]
    right_wrist_y = keypoints[10][1]
    
    # Simple Logic: Are wrists above the nose? (Y coordinates decrease upwards)
    if left_wrist_y < nose_y and right_wrist_y < nose_y:
        return "Hands Up!"
    elif left_wrist_y < nose_y:
        return "Left Hand Up"
    elif right_wrist_y < nose_y:
        return "Right Hand Up"
    else:
        return "Neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # YOLOv8 Inference
    results = model(frame, verbose=False)
    
    # Visualize results on the frame
    annotated_frame = results[0].plot()
    
    # Extract keypoints for logic
    # results[0].keypoints.data is a Tensor (Batch, Points, 3)
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        # Get keypoints for the first detected person
        person_kps = results[0].keypoints.data[0].cpu().numpy()
        
        gesture = detect_gesture(person_kps)
        
        cv2.putText(annotated_frame, gesture, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('YOLOv8 Pose Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()