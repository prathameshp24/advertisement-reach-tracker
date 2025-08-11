import cv2
import numpy as np
from ultralytics import YOLO
import dlib

class ImprovedGazeDetector:
    def __init__(self):
        # Load face detection model
        self.face_model = YOLO("yolov8n-face.pt")
        
        # Load facial landmark predictor
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Gaze detection parameters
        self.gaze_threshold = 35  # Increased from 30 to be more lenient
        self.horizontal_gaze_threshold = 180
        
        # For debug visualization
        self.show_landmarks = True
        self.show_angles = True
        
        # Focus tracking - based on frames instead of real time
        self.frames_to_focus = 30  # Assuming 10fps, this is 3 seconds
        self.face_tracking = {}  # Track faces and their gaze status

        # Critical landmark indices to display
        # Eyes: 36-47, Jaw line: 0-16
        self.critical_landmarks = list(range(36, 48)) + list(range(0, 17))
        
        # Face tracking
        self.face_tracking_tolerance = 50  # pixel distance for considering the same face
        
        # Face visibility threshold - percentage of landmarks that must be visible
        self.face_visibility_threshold = 0.8  # 80% of landmarks must be visible
        
        # Relative positioning thresholds for eyes-nose-chin alignment
        self.z_depth_threshold = 15.0  # Threshold for z-depth difference in mm

    def calculate_head_pose(self, landmarks, frame_shape):
        """Calculate head pose angles with more robust checks"""
        # 3D model points (generic head model)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)

        # 2D image points from landmarks
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left mouth corner
            landmarks[54]   # Right mouth corner
        ], dtype=np.float64)

        # Camera matrix approximation
        focal_length = frame_shape[1]
        center = (frame_shape[1]/2, frame_shape[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if not success:
            return None, None, None, None
        
        # Project 3D points to 2D for additional processing
        nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), 
                                                rotation_vector, translation_vector, 
                                                camera_matrix, dist_coeffs)
        
        nose_end_point2D = tuple(map(int, nose_end_point2D[0][0]))

        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        pitch, yaw, roll = angles
        
        return pitch, yaw, roll, translation_vector

    def check_face_visibility(self, landmarks, frame_shape):
        """Check if the face is fully visible in the frame"""
        # Count landmarks that are within frame boundaries
        h, w = frame_shape[:2]
        valid_landmarks = 0
        
        for (x, y) in landmarks:
            if 0 <= x < w and 0 <= y < h:
                valid_landmarks += 1
        
        visibility_ratio = valid_landmarks / len(landmarks)
        return visibility_ratio >= self.face_visibility_threshold

    def check_relative_face_positioning(self, landmarks, translation_vector):
        """Check if eyes, nose, and chin are positioned correctly for camera gaze
        
        When looking at camera:
        - Eyes should be further back (larger Z) than nose
        - Nose should be further back than chin
        """
        # Get the average eye position (center between eyes)
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        eye_center = np.mean([left_eye_center, right_eye_center], axis=0)
        
        # Get nose tip and chin positions
        nose_tip = landmarks[30]
        chin = landmarks[8]
        
        # Use the translation vector to estimate relative Z positions
        # This is an approximation based on the head pose
        z_value = translation_vector[2][0]
        
        # Calculate approximate Z depth of key features based on pose
        # When looking up at camera, eyes are further back than nose, and nose further back than chin
        eye_z = z_value - 20  # Eyes are typically set back in the face
        nose_z = z_value      # Nose is our reference point
        chin_z = z_value + 20 # Chin projects forward
        
        # Check if the relative positions match a "looking at camera" configuration
        relative_positioning_correct = (eye_z < nose_z) and (nose_z < chin_z)
        
        # For debug purposes
        z_depth_info = {
            "eye_z": eye_z,
            "nose_z": nose_z,
            "chin_z": chin_z,
            "aligned": relative_positioning_correct
        }
        
        return relative_positioning_correct, z_depth_info

    def find_matching_face_id(self, face_bbox):
        """Find if this detected face matches any previously tracked face"""
        x1, y1, x2, y2 = face_bbox
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        for face_id, data in self.face_tracking.items():
            coords = list(map(int, face_id.split('_')))
            old_center = ((coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2)
            
            # Calculate distance between face centers
            distance = np.sqrt((face_center[0] - old_center[0])**2 + (face_center[1] - old_center[1])**2)
            
            if distance < self.face_tracking_tolerance:
                return face_id
                
        return None

    def process_frame(self, frame):
        active_faces = set()
        
        # Face detection
        face_results = self.face_model(frame, verbose=False)[0]
        faces = [box.xyxy[0].cpu().numpy().astype(int) for box in face_results.boxes]
        
        for (x1, y1, x2, y2) in faces:
            # Try to match with a previously tracked face
            face_id = self.find_matching_face_id((x1, y1, x2, y2))
            
            if not face_id:
                face_id = f"{x1}_{y1}_{x2}_{y2}"
                # Initialize new face tracking data
                self.face_tracking[face_id] = {
                    "gaze_frames": 0,
                    "is_focused": False,
                    "frame_count": 0
                }
            
            active_faces.add(face_id)
            self.face_tracking[face_id]["frame_count"] += 1
            
            dlib_rect = dlib.rectangle(x1, y1, x2, y2)
            
            # Convert to grayscale for landmark detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            try:
                # Get facial landmarks
                shape = self.predictor(gray, dlib_rect)
                landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float64)
                
                # Check if face is fully visible
                face_visible = self.check_face_visibility(landmarks, frame.shape)
                
                # Calculate head pose
                pitch, yaw, roll, translation_vector = self.calculate_head_pose(landmarks, frame.shape)
                
                if pitch is None:
                    continue
                
                # Check relative positioning of eyes, nose, and chin
                relative_positioning_correct, z_depth_info = self.check_relative_face_positioning(
                    landmarks, translation_vector)
                
                # Determine if person is gazing at camera based on multiple criteria
                # If full face is visible, we use both angle thresholds and relative positioning
                # If face is partially visible, we rely more on relative positioning
                if face_visible:
                    is_gazing = (abs(yaw) < self.gaze_threshold and 
                                -180 <= pitch <= 180 and 
                                relative_positioning_correct)
                else:
                    # For partially visible faces, rely more on relative positioning
                    is_gazing = relative_positioning_correct
                
                face_data = self.face_tracking[face_id]
                
                # Update gaze tracking logic - frame-based instead of time-based
                if is_gazing:
                    # Increment gaze frame counter
                    face_data["gaze_frames"] += 1
                    
                    # Check if we've reached the threshold
                    if face_data["gaze_frames"] >= self.frames_to_focus:
                        face_data["is_focused"] = True
                else:
                    # Reset counter if not gazing
                    face_data["gaze_frames"] = 0
                    face_data["is_focused"] = False
                
                # Determine seconds remaining until focus
                # Calculate with integers to get 3, 2, 1 countdown
                if is_gazing and not face_data["is_focused"]:
                    frames_remaining = self.frames_to_focus - face_data["gaze_frames"]
                    seconds_remaining = max(1, int(frames_remaining / 10) + 1)  # +1 because we want to show 3,2,1 not 2,1,0
                else:
                    seconds_remaining = 3
                
                # Determine status and display color
                if face_data["is_focused"]:
                    # Yellow for focused
                    color = (0, 255, 255)  
                    status = "FOCUSED"
                elif is_gazing:
                    # Green for gazing
                    color = (0, 255, 0)
                    status = f"GAZING ({seconds_remaining}s to focus)"
                else:
                    # Red for not gazing
                    color = (0, 0, 255)
                    status = "NOT GAZING"
                
                # Draw face rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Display gaze status
                cv2.putText(frame, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display angles on top of each bounding box
                angle_text = f"Y:{yaw:.1f}° P:{pitch:.1f}° R:{roll:.1f}°"
                cv2.putText(frame, angle_text, (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Display relative positioning information
                z_text = f"Aligned: {z_depth_info['aligned']} Vis: {face_visible:.1f}"
                cv2.putText(frame, z_text, (x1, y1-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw only critical facial landmarks if enabled
                if self.show_landmarks:
                    for idx, (x, y) in enumerate(landmarks.astype(int)):
                        if idx in self.critical_landmarks:
                            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                    
                    # Draw lines to visualize the face alignment
                    # Eyes to nose
                    left_eye = landmarks[36].astype(int)
                    right_eye = landmarks[45].astype(int)
                    nose = landmarks[30].astype(int)
                    chin = landmarks[8].astype(int)
                    
                    cv2.line(frame, tuple(left_eye), tuple(nose), (0, 255, 255), 1)
                    cv2.line(frame, tuple(right_eye), tuple(nose), (0, 255, 255), 1)
                    cv2.line(frame, tuple(nose), tuple(chin), (0, 255, 255), 1)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Clean up faces that haven't been seen in recent frames
        faces_to_remove = []
        for face_id, data in self.face_tracking.items():
            if face_id not in active_faces:
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            self.face_tracking.pop(face_id, None)
                
        return frame

def test_gaze_detection(video_path):
    detector = ImprovedGazeDetector()
    cap = cv2.VideoCapture(video_path if video_path else 0)  # Use webcam if no video
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Adjust frames to focus based on actual video FPS
    if fps > 0:
        detector.frames_to_focus = int(1 * fps)  # 3 seconds at video's actual FPS
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter('audit1.mp4', fourcc, fps, (frame_width, frame_height))
    
    print(f"Starting enhanced gaze detection with relative face positioning...")
    print(f"Video FPS: {fps}, Frames required for focus: {detector.frames_to_focus}")
    print("Press 'q' to quit, 'l' to toggle landmarks, 'a' to toggle angles")
    print("Saving annotated video to 'output_relative_face_position.mp4'")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
        
        # Display results
        cv2.imshow("Enhanced Gaze Detection", processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            detector.show_landmarks = not detector.show_landmarks
        elif key == ord('a'):
            detector.show_angles = not detector.show_angles
            
    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "video.mp4"  # Leave empty for webcam or specify path
    test_gaze_detection(video_path)