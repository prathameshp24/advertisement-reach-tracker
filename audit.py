import cv2
import numpy as np
import os
from collections import deque
from ultralytics import YOLO
import csv
import argparse
from pathlib import Path

class FaceDetectionAuditor:
    def __init__(self, fps=30, output_dir="audit_output"):
        # Create output directory
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Face detection model
        self.face_model = YOLO("yolov8n-face.pt")
        
        # Gender and Age models
        self.gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
        self.age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.gender_list = ['Male', 'Female']
        self.age_ranges = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
        self.age_groups = ['child', 'child', 'child', 'young adult', 'adult', 'adult', 'old', 'old']
        
        # Face tracking parameters
        self.tracked_faces = {}  # {face_id: {'last_bbox': (x1,y1,x2,y2)}}
        self.max_face_distance = 50  # pixels for considering same face
        
        # CSV for audit data
        self.csv_path = os.path.join(output_dir, "audit_results.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, 
                                         fieldnames=['image_id', 'number_of_persons', 'number_of_faces', 
                                                    'number_of_males', 'number_of_females'])
        self.csv_writer.writeheader()

    def _get_face_id(self, current_bbox, current_centroid):
        """Match current face to existing tracked faces using centroid distance"""
        for face_id, data in self.tracked_faces.items():
            last_centroid = ((data['last_bbox'][0] + data['last_bbox'][2])/2,
                             (data['last_bbox'][1] + data['last_bbox'][3])/2)
            distance = np.linalg.norm(np.array(current_centroid) - np.array(last_centroid))
            if distance < self.max_face_distance:
                return face_id
        return None

    def process_frame(self, frame, frame_number):
        current_faces = {}
        
        # Create image ID for this frame
        image_id = f"image_{frame_number:06d}"
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        
        # Save the original frame
        cv2.imwrite(image_path, frame)
        
        # Detect faces using YOLO
        face_results = self.face_model(frame, verbose=False)[0]
        faces = [box.xyxy[0].cpu().numpy().astype(int) for box in face_results.boxes]
        
        # Count actual number of people (may be different from faces if YOLO detects partial faces)
        number_of_faces = len(faces)
        number_of_persons = number_of_faces  # In most cases, these will be the same
        males = 0
        females = 0
        
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        for (x1, y1, x2, y2) in faces:
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            
            try:
                # Get face ROI for gender and age detection
                face_roi = frame[y1:y2, x1:x2]
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                
                # Predict gender
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender_idx = np.argmax(gender_preds)
                gender = self.gender_list[gender_idx]
                
                # Update gender counts
                if gender == 'Male':
                    males += 1
                else:
                    females += 1
                
                # Predict age
                self.age_net.setInput(blob)
                age_idx = np.argmax(self.age_net.forward())
                age_group = self.age_groups[age_idx]
                age_range = self.age_ranges[age_idx]
                
                # Track face
                centroid = ((x1 + x2)/2, (y1 + y2)/2)
                face_id = self._get_face_id((x1, y1, x2, y2), centroid)
                
                if face_id is None:
                    face_id = f"face_{len(self.tracked_faces)+1}"
                    self.tracked_faces[face_id] = {
                        'last_bbox': (x1, y1, x2, y2)
                    }
                
                self.tracked_faces[face_id]['last_bbox'] = (x1, y1, x2, y2)
                
                # Draw rectangle and text on visualization frame
                color = (0, 255, 0)  # Green color for all faces
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Display gender and age
                status = f"{gender}, {age_group} ({age_range})"
                cv2.putText(vis_frame, status, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Store face data
                current_faces[face_id] = {
                    'gender': gender,
                    'age_group': age_group,
                    'age_range': age_range
                }
                
            except Exception as e:
                print(f"Error processing face in frame {frame_number}: {e}")
                continue
        
        # Clean up tracked faces that are no longer visible
        for face_id in list(self.tracked_faces.keys()):
            if face_id not in current_faces:
                del self.tracked_faces[face_id]
        
        # Write audit data to CSV
        self.csv_writer.writerow({
            'image_id': image_id,
            'number_of_persons': number_of_persons,
            'number_of_faces': number_of_faces,
            'number_of_males': males,
            'number_of_females': females
        })
        
        # Save annotated frame
        annotated_path = os.path.join(self.images_dir, f"{image_id}_annotated.jpg")
        cv2.imwrite(annotated_path, vis_frame)
        
        # Add frame info to visualization frame
        stats_text = f"Frame: {frame_number} | Faces: {number_of_faces} | M: {males} F: {females}"
        cv2.putText(vis_frame, stats_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame, current_faces
    
    def close(self):
        """Close CSV file and clean up"""
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
            print(f"Audit results saved to {self.csv_path}")

def run_audit(video_path, output_dir="audit_output", save_every=1, display=True):
    """
    Run the face detection audit on a video
    
    Args:
        video_path: Path to video file or 0 for webcam
        output_dir: Directory to save audit results
        save_every: Save every nth frame (1 = save all frames)
        display: Whether to display the video during processing
    """
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create auditor
    auditor = FaceDetectionAuditor(fps, output_dir)
    
    # Set up video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, "audit_video.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Starting audit of {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Output directory: {output_dir}")
    print(f"Saving every {save_every} frame(s)")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame if it's one we want to save
        if frame_count % save_every == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
            processed_frame, _ = auditor.process_frame(frame, frame_count)
            out.write(processed_frame)
            
            if display:
                cv2.imshow("Face Detection Audit", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Clean up
    auditor.close()
    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"Audit complete. Processed {frame_count} frames.")
    print(f"Audit video saved to {out_path}")
    print(f"Frames saved to {os.path.join(output_dir, 'frames')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection Audit Tool")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (default: use webcam)")
    parser.add_argument("--output", type=str, default="audit_output", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1, help="Save every nth frame")
    parser.add_argument("--no_display", action="store_true", help="Don't display video during processing")
    
    args = parser.parse_args()
    
    video_path = args.video
    if video_path is None:
        video_path = "4bc63a9f-3c41-4e23-9d7f-39ff1b61fe66_primary_1736913992974.mp4" 
    
    run_audit(video_path, args.output, args.save_every, not args.no_display)