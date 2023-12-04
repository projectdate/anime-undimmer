import argparse
from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt

def multiply_colors(frame, factor):
    """
    Multiply each color in the frame by a given factor.
    Clipping is performed to ensure pixel values stay within valid range.
    """
    
    return np.clip(frame * factor, 0, 255).astype('uint8')

def process_video(input_file, output_file, factor):
    """
    Process the video, multiplying each frame's pixel values by the specified factor.
    """
    clip = VideoFileClip(input_file)
    modified_clip = clip.fl_image(lambda frame: multiply_colors(frame, factor))
    modified_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

def plot_max_values(input_file):
    """
    Plot max frame value for each half second throughout the video.
    """
    clip = VideoFileClip(input_file)
    # Store the max value of frame seen every quarter second
    print("Starting analysis...")
    max_values = []
    import os
    import pickle
    
    # Define the cache file path
    cache_file = f"{input_file}_max_values.pkl"
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        # Load the max values from the cache file
        with open(cache_file, 'rb') as f:
            max_values = pickle.load(f)
    else:
        # Calculate the max values and store them in the cache file
        for i, (t, frame) in enumerate(clip.iter_frames(with_times=True)):
            if i % 6 == 0:  # Check every 1/4 of a second, so every 6th frame
                max_values.append(np.max(frame))
        with open(cache_file, 'wb') as f:
            pickle.dump(max_values, f)
            
    print(max_values)
    plt.plot([x/4 for x in range(len(max_values))], max_values)
    plt.title('Max frame value per quarter second')
    plt.xlim(0, len(max_values)/4)  # Set x-axis range to match the number of data points in seconds
    plt.xticks(np.arange(0, len(max_values)/4, 5))  # Show ticks on the x axis every 5 seconds
    
    # Find all the time ranges where at least 3 consecutive values are below 180
    below_180_ranges = []
    count = 0
    start_time = 0
    for i, value in enumerate(max_values):
        if value < 180:
            if count == 0:
                start_time = i / 4  # Convert frame index to time in seconds
            count += 1
        else:
            if count >= 3:
                below_180_ranges.append((start_time, i / 4))  # Add the start and end time of the range
            count = 0
    if count >= 3:
        below_180_ranges.append((start_time, len(max_values) / 4))  # Add the last range if it ends at the end of the video
    
    # Print all the time ranges where at least 3 consecutive values are below 180
    for start, end in below_180_ranges:
        print(f"Time range with at least 3 consecutive values below 180: {start} - {end} seconds")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Multiply color values in a video by a factor.')
    parser.add_argument('input_file', type=str, help='Path to the input video file')
    parser.add_argument('output_file', type=str, help='Path to the output video file')
    parser.add_argument('factor', type=float, help='Factor to multiply color values by')
    parser.add_argument('--only_plot', action='store_true', help='Only plot max frame value for each quarter second')

    args = parser.parse_args()

    plot_max_values(args.input_file)
    if (not args.only_plot):
        process_video(args.input_file, args.output_file, args.factor)

if __name__ == "__main__":
    main()