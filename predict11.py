import numpy as np
from ultralytics import YOLO
from collections import OrderedDict, defaultdict, deque
import dlib
from deepface import DeepFace
import os
import psycopg2
from psycopg2.extras import execute_batch
import time
import re
from datetime import datetime
import cv2

class PerformanceMonitor:
    def __init__(self):
        self.timings = defaultdict(list)
        
    def record_time(self, operation, elapsed_time):
        self.timings[operation].append(elapsed_time)
    
    def get_summary(self):
        summary = {}
        for operation, times in self.timings.items():
            summary[operation] = {
                'total_time': sum(times),
                'average_time': sum(times) / len(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'calls': len(times)
            }
        return summary
    
    def print_summary(self):
        summary = self.get_summary()
        print("\nPerformance Summary:")
        print("=" * 80)
        print(f"{'Operation':<30} {'Total(s)':<10} {'Avg(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10} {'Calls':<10}")
        print("-" * 80)
        
        for operation, stats in summary.items():
            print(f"{operation:<30} "
                  f"{stats['total_time']:>9.2f} "
                  f"{(stats['average_time']*1000):>9.2f} "
                  f"{(stats['min_time']*1000):>9.2f} "
                  f"{(stats['max_time']*1000):>9.2f} "
                  f"{stats['calls']:>9}")
        print("=" * 80)

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="",
            user="",
            password="",
            host="",
            port=""
        )
        self.cursor = self.conn.cursor()
        print("Database connection established successfully")
        
    def insert_summary(self, summary_data):
        try:
            query = """
            INSERT INTO image_proc_summary 
            (device_id, feed_time, process_time_stamp, people_count, 
             male_count, female_count, created_date, comments, description, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING img_proc_id;
            """
            feed_time = int(summary_data['feed_time'].timestamp() * 1000) if isinstance(summary_data['feed_time'], datetime) else summary_data['feed_time']
            
            self.cursor.execute(query, (
                summary_data['device_id'],
                feed_time,
                summary_data['process_time_stamp'],
                summary_data['people_count'],
                summary_data['male_count'],
                summary_data['female_count'],
                summary_data['created_date'],
                summary_data['comments'],
                summary_data['description'],
                summary_data.get('image_path')
            ))
            img_proc_id = self.cursor.fetchone()[0]
            self.conn.commit()
            print(f"Summary inserted successfully. img_proc_id: {img_proc_id}")
            return img_proc_id
        except Exception as e:
            print(f"Error inserting summary: {e}")
            self.conn.rollback()
            raise
        
    def insert_details(self, details_data, img_proc_id):
        try:
            query = """
            INSERT INTO image_proc_details 
            (device_id, feed_time, image_process_time_stamp, img_proc_id,
             person_id, gazing, gender, age_group, sentiment, created_date, comments, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            prepared_data = [
                (detail['device_id'],
                 int(detail['feed_time'].timestamp() * 1000) if isinstance(detail['feed_time'], datetime) else detail['feed_time'],
                 detail['image_process_time_stamp'],
                 img_proc_id,
                 detail['person_id'],
                 detail['gazing'],
                 detail['gender'],
                 detail['age_group'],
                 detail['sentiment'],
                 detail['created_date'],
                 detail['comments'],
                 detail['description'])
                for detail in details_data
            ]
            
            execute_batch(self.cursor, query, prepared_data)
            self.conn.commit()
            print(f"Details inserted successfully for img_proc_id: {img_proc_id}")
        except Exception as e:
            print(f"Error inserting details: {e}")
            self.conn.rollback()
            raise

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
            print("Database connection closed successfully")
        except Exception as e:
            print(f"Error closing database connection: {e}")

class DetectionProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.device_id = self._extract_device_id(video_path)
        self.feed_time = self._extract_feed_time(video_path)
        self.base_output_path = "processed_frames"
        self.output_dir = self._create_output_directory()
        
        # Initialize models
        print("Loading models...")
        start_time = time.time()
        self.person_model = YOLO("yolov8n.pt")
        self.face_model = YOLO("yolov8n-face.pt")
        self.gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
        self.age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print(f"Models loaded in {time.time()-start_time:.2f} seconds")
        
        # Configuration
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.gender_list = ['Male', 'Female']
        self.age_groups = ['child', 'child', 'child', 'young adult', 'adult', 'adult', 'old', 'old']
        self.max_face_distance = 50  # pixels
        
        # Gaze detection parameters - updated from second snippet
        self.gaze_threshold = 35  # Increased from 30 to be more lenient
        self.horizontal_gaze_threshold = 180
        
        # Debug visualization
        self.show_landmarks = True
        self.show_angles = True

        # Focus tracking
        self.frames_to_focus = 30
        self.face_tracking = {}

        # Critical landmark indices to display
        # Eyes: 36-47, Jaw line: 0-16
        self.critical_landmarks = list(range(36, 48)) + list(range(0, 17))
        
        # Face tracking
        self.face_tracking_tolerance = 50  # pixel distance for considering the same face
        
        # Face visibility threshold - percentage of landmarks that must be visible
        self.face_visibility_threshold = 0.8  # 80% of landmarks must be visible
        
        # Relative positioning thresholds for eyes-nose-chin alignment
        self.z_depth_threshold = 15.0  # Threshold for z-depth difference in mm
        
        # Tracking
        self.tracked_faces = OrderedDict()
        self.next_face_id = 0
        self.perf_monitor = PerformanceMonitor()
        self.db = DatabaseManager()

    def _create_output_directory(self):
        """Create directory structure in the shared mount"""
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_dir = os.path.join(
            self.base_output_path,
            datetime.now().strftime("%Y-%m-%d"),  # Optional date organization
            video_name
        )
            
        try:
            # Create directories with 775 permissions (rwxrwxr-x)
            os.makedirs(output_dir, exist_ok=True, mode=0o775)
            print(f"Created output directory at: {output_dir}")
            return output_dir
        except PermissionError as pe:
            print(f"Permission denied: {pe}")
            print(f"Please ensure you have write access to: {self.base_output_path}")
            print("Try: sudo chmod -R 775 /mnt/smb_mount/prathamesh/tarneaProject/processed_frames")
            raise
        except Exception as e:
            print(f"Error creating directory: {e}")
            raise

    def _extract_device_id(self, video_path):
        match = re.search(r'(.*?)_primary_', os.path.basename(video_path))
        return match.group(1) if match else None
        
    def _extract_feed_time(self, video_path):
        match = re.search(r'_primary_(\d+)', video_path)
        return int(match.group(1)) if match else None

    def save_processed_frame(self, frame, img_proc_id):
        """Save frame as /home/processed_frame/video_name/img_proc_id.jpg"""
        try:
            # Generate the exact filename as requested
            filename = f"{img_proc_id}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Compress to under 100KB
            quality = 85  # Start with high quality
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Check and adjust file size if needed
            file_size_kb = os.path.getsize(filepath) / 1024
            if file_size_kb > 100:
                quality = 70  # Reduce quality
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                file_size_kb = os.path.getsize(filepath) / 1024
                if file_size_kb > 100:
                    # If still too large, resize
                    h, w = frame.shape[:2]
                    scale = (100 * 1024 / (file_size_kb * 1024)) ** 0.5
                    new_size = (int(w * scale), int(h * scale))
                    resized = cv2.resize(frame, new_size)
                    cv2.imwrite(filepath, resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            print(f"Frame saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving frame: {e}")
            return None

    def calculate_head_pose(self, landmarks, frame_shape):
        """Calculate head pose angles with more robust checks - from the improved gaze detector"""
        try:
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
            
        except Exception as e:
            print(f"Error in head pose calculation: {e}")
            return None, None, None, None

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

    def detect_demographics(self, face_roi):
        try:
            # Preprocess face for gender and age detection
            blob = cv2.dnn.blobFromImage(
                face_roi, 1.0, (227, 227), 
                self.MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender detection
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Age detection
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_group = self.age_groups[age_preds[0].argmax()]
            
            return gender, age_group
        except Exception as e:
            print(f"Error in demographics detection: {e}")
            return "Unknown", "Unknown"

    def find_matching_face_id(self, face_bbox):
        """Find if this detected face matches any previously tracked face"""
        x1, y1, x2, y2 = face_bbox
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Find closest existing face
        min_dist = float('inf')
        matched_id = None
        
        for face_id, data in self.face_tracking.items():
            coords = list(map(int, face_id.split('_')))
            old_center = ((coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2)
            
            # Calculate distance between face centers
            distance = np.sqrt((face_center[0] - old_center[0])**2 + (face_center[1] - old_center[1])**2)
            
            if distance < self.face_tracking_tolerance and distance < min_dist:
                min_dist = distance
                matched_id = face_id
                
        return matched_id

    def _match_or_create_face_id(self, centroid):
        # Find closest existing face
        min_dist = float('inf')
        matched_id = None
        
        for face_id, data in self.tracked_faces.items():
            dist = np.linalg.norm(np.array(centroid) - np.array(data['centroid']))
            if dist < self.max_face_distance and dist < min_dist:
                min_dist = dist
                matched_id = face_id
        
        # If no match found, create new ID
        if matched_id is None:
            matched_id = self.next_face_id
            self.next_face_id += 1
            self.tracked_faces[matched_id] = {'centroid': centroid}
        
        # Update centroid for the matched face
        self.tracked_faces[matched_id]['centroid'] = centroid
        return matched_id

    def process_frame(self, frame, frame_number, fps):
        frame_start_time = time.time()
        current_time = datetime.now()
        active_faces = set()
        
        # Calculate timestamp based on frame number and FPS
        if isinstance(self.feed_time, int):
            epoch_ms = self.feed_time + (frame_number / fps) * 1000
            process_timestamp = datetime.fromtimestamp(epoch_ms / 1000)
        else:
            process_timestamp = current_time
        
        # Person detection
        person_start = time.time()
        person_results = self.person_model(frame)[0]
        people_count = len([box for box in person_results.boxes if box.cls == 0])
        self.perf_monitor.record_time('Person Detection', time.time() - person_start)
        
        # Face detection
        face_start = time.time()
        face_results = self.face_model(frame, conf=0.4)[0]
        current_faces = [box.xyxy[0].cpu().numpy().astype(int) for box in face_results.boxes]
        self.perf_monitor.record_time('Face Detection', time.time() - face_start)
        
        # Process each face
        face_details = []
        male_count = 0
        female_count = 0
        
        # Convert to grayscale once for all face processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for (x1, y1, x2, y2) in current_faces:
            try:
                # Try to match with a previously tracked face
                face_id_str = f"{x1}_{y1}_{x2}_{y2}"
                face_id_match = self.find_matching_face_id((x1, y1, x2, y2))
                
                if not face_id_match:
                    # Initialize new face tracking data
                    self.face_tracking[face_id_str] = {
                        "gaze_frames": 0,
                        "is_focused": False,
                        "frame_count": 0
                    }
                else:
                    face_id_str = face_id_match
                
                active_faces.add(face_id_str)
                
                if face_id_str in self.face_tracking:
                    self.face_tracking[face_id_str]["frame_count"] += 1
                
                # Crop face ROI
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                # Detect gender and age
                gender, age_group = self.detect_demographics(face_roi)
                if gender == 'Male':
                    male_count += 1
                else:
                    female_count += 1
                
                # IMPROVED GAZE DETECTION INTEGRATION
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                
                # Get facial landmarks
                shape = self.predictor(gray, dlib_rect)
                landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float64)
                
                # Check if face is fully visible
                face_visible = self.check_face_visibility(landmarks, frame.shape)
                
                # Calculate head pose with improved method
                pitch, yaw, roll, translation_vector = self.calculate_head_pose(landmarks, frame.shape)
                
                if pitch is None:
                    continue
                
                # Check relative positioning of eyes, nose, and chin
                relative_positioning_correct, z_depth_info = self.check_relative_face_positioning(
                    landmarks, translation_vector)
                
                # Determine if person is gazing at camera based on multiple criteria
                if face_visible:
                    is_gazing = (abs(yaw) < self.gaze_threshold and 
                                -180 <= pitch <= 180 and 
                                relative_positioning_correct)
                else:
                    # For partially visible faces, rely more on relative positioning
                    is_gazing = relative_positioning_correct
                
                face_data = self.face_tracking[face_id_str]
                
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
                if is_gazing and not face_data["is_focused"]:
                    frames_remaining = self.frames_to_focus - face_data["gaze_frames"]
                    seconds_remaining = max(1, int(frames_remaining / 10) + 1)
                else:
                    seconds_remaining = 3
                
                # Determine status and display color for annotations
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
                
                # ANNOTATIONS FOR VISUALIZATION
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
                
                # Display gender and age
                demo_text = f"{gender}, {age_group}"
                cv2.putText(frame, demo_text, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
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
                
                # Assign a numerical unique ID for the database using the original method
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                person_id = self._match_or_create_face_id(centroid)
                
                # Prepare face details for database
                face_details.append({
                    'device_id': self.device_id,
                    'feed_time': self.feed_time,
                    'image_process_time_stamp': process_timestamp,
                    'person_id': person_id,
                    'gazing': is_gazing,
                    'gender': gender,
                    'age_group': age_group,
                    'sentiment': 'normal',
                    'created_date': current_time,
                    'comments': None,
                    'description': None
                })
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Clean up faces that haven't been seen in recent frames
        faces_to_remove = []
        for face_id_str in self.face_tracking:
            if face_id_str not in active_faces:
                faces_to_remove.append(face_id_str)
        
        for face_id_str in faces_to_remove:
            self.face_tracking.pop(face_id_str, None)
        
        # Prepare summary
        summary = {
            'device_id': self.device_id,
            'feed_time': self.feed_time,
            'process_time_stamp': process_timestamp,
            'people_count': people_count,
            'male_count': male_count,
            'female_count': female_count,
            'created_date': current_time,
            'comments': None,
            'description': None,
            'image_path': None # Will be filled after saving frame
        }
        
        self.perf_monitor.record_time('Total Frame Processing', time.time() - frame_start_time)
        return summary, face_details, frame

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video file {self.video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        processed_frames = 0
        
        # Adjust frames to focus based on actual video FPS
        if fps > 0:
            self.frames_to_focus = int(3 * fps)  # 3 seconds at video's actual FPS
        
        print(f"Video FPS: {fps}, Frames required for focus: {self.frames_to_focus}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every second (or adjust as needed)
            if frame_count % (fps // 2) == 0:
                summary, details, annotated_frame = self.process_frame(frame, frame_count, fps)
                
                # Only process frames with detected people
                if summary['people_count'] > 0:
                    processed_frames += 1
                    print(f"Frame {frame_count}: {summary['people_count']} people, {summary['male_count']} males, {summary['female_count']} females")
                    
                    # Database operations
                    db_start = time.time()
                    try:
                        img_proc_id = self.db.insert_summary(summary)
                        if details:
                            self.db.insert_details(details, img_proc_id)
                        
                        # Save and update frame path - using annotated frame with visualizations
                        frame_path = self.save_processed_frame(annotated_frame, img_proc_id)
                        if frame_path:
                            self.db.cursor.execute(
                                "UPDATE image_proc_summary SET image_path = %s WHERE img_proc_id = %s",
                                (frame_path, img_proc_id)
                            )
                            self.db.conn.commit()
                    except Exception as e:
                        print(f"Database error: {e}")
                        self.db.conn.rollback()
                    
                    self.perf_monitor.record_time('Database Operations', time.time() - db_start)
            
            frame_count += 1
        
        cap.release()
        self.db.close()
        
        # Print performance summary
        total_time = sum(t['total_time'] for t in self.perf_monitor.get_summary().values())
        print(f"\nProcessing complete. Processed {processed_frames} frames in {total_time:.2f} seconds")
        self.perf_monitor.print_summary()


def process_single_video(video_path):
    """
    This is the function that monitor.py will call.
    It creates an instance of DetectionProcessor and processes the video.
    """
    try:
        # Create processor instance
        processor = DetectionProcessor(video_path)
        # Process the video
        processor.process_video()
        return True
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return False