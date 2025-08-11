import cv2
import os

def extract_frames(video_path, output_folder):
    """
    Extract all frames from a video file and save them as JPG images.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to the folder where frames will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Total frames in video: {total_frames}")
    print(f"Starting frame extraction to {output_folder}...")
    
    # Read and save each frame
    while True:
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Save the frame as a JPG file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # Increment frame counter
        frame_count += 1
        
        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    # Release the video capture object
    cap.release()
    
    print(f"Frame extraction complete. Extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    # Path to the output video created by the gaze detection script
    video_path = "audit1.mp4"
    
    # Folder where frames will be saved
    output_folder = "audit_gaze"
    
    # Extract frames
    extract_frames(video_path, output_folder)