from moviepy.editor import VideoFileClip
import os
import cv2
import matplotlib.pyplot as plt

def convert_mp4_to_png(video_path, output_folder):
    # Load the video clip
    clip = VideoFileClip(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each frame in the video
    for i, frame in enumerate(clip.iter_frames()):
        #downscale the image to 480x640
        frame = cv2.resize(frame, (640, 480))
        # Save the frame as a PNG image
        frame_path = os.path.join(output_folder, f"frame_{i}.png")
        plt.imsave(frame_path, frame)





        
    # Close the video clip
    clip.close()

# Example usage
video_path = r"C:\Users\gj1182\Documents\GitHub\IHS_projekt\data_rv_ljudje\20240501_191641.mp4"
output_folder = r"C:\Users\gj1182\Documents\GitHub\IHS_projekt\data_rv_ljudje\png"
convert_mp4_to_png(video_path, output_folder)