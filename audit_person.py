import cv2
import csv
import argparse
from ultralytics import YOLO
from datetime import datetime

def process_video(video_path, model_path, output_csv=None, display=True, confidence=0.25):
    """
    Process video frames, count persons and save results to CSV.
    
    Args:
        video_path (str): Path to the video file
        model_path (str): Path to the YOLOv8 model weights
        output_csv (str, optional): Path to save the CSV file. If None, generates a timestamp-based name.
        display (bool): Whether to display the processed video
        confidence (float): Confidence threshold for detections
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create CSV file name with timestamp if not provided
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"person_count_{timestamp}.csv"
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Frame Number', 'Timestamp (s)', 'Person Count'])
        
        # Process frames
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Increment frame counter
            frame_number += 1
            
            # Calculate timestamp
            timestamp = frame_number / fps
            
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=confidence)
            
            # Get the detection results
            boxes = results[0].boxes
            
            # Count persons (class_id 0 in COCO dataset is 'person')
            person_count = sum(1 for box in boxes if box.cls.cpu().numpy()[0] == 0)
            
            # Write to CSV
            csvwriter.writerow([frame_number, f"{timestamp:.2f}", person_count])
            
            # Display the frame with detection count if requested
            if display:
                # Annotate frame with person count
                annotated_frame = results[0].plot()
                cv2.putText(
                    annotated_frame, 
                    f"Persons: {person_count}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Display progress
                progress = frame_number / frame_count * 100
                print(f"Processing: {progress:.1f}% - Frame {frame_number}/{frame_count} - Persons: {person_count}", end='\r')
                
                # Show the frame
                cv2.imshow("Person Detection", annotated_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Release resources
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"\nDone! Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count persons in video frames using YOLOv8")
    parser.add_argument("video_path", help="path/video/mp4")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 model weights (default: yolov8n.pt)")
    parser.add_argument("--output", help="Path to save the CSV file (default: auto-generated)")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections (default: 0.25)")
    
    args = parser.parse_args()
    
    process_video(
        args.video_path,
        args.model,
        args.output,
        not args.no_display,
        args.conf
    )