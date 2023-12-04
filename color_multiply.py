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

def calculate_epilepsy_risk(frame_values):
    # Calculate the difference between consecutive frames
    frame_diffs = np.diff(frame_values, axis=0)
    # Calculate the absolute sum of pixel differences for each frame difference
    abs_sum_diffs = np.sum(np.abs(frame_diffs), axis=(1, 2, 3))
    # Calculate the mean of the absolute sum of differences
    mean_abs_sum_diff = np.mean(abs_sum_diffs)
    # Calculate the standard deviation of the absolute sum of differences
    std_abs_sum_diff = np.std(abs_sum_diffs)
    # Print the mean and standard deviation of the absolute sum of differences
    # print(f"Mean absolute sum of differences between consecutive frames: {mean_abs_sum_diff}")
    print(f"Standard deviation of absolute sum of differences between consecutive frames: {std_abs_sum_diff}")
    # If the standard deviation is high, the video is more likely to cause epilepsy
    return std_abs_sum_diff

def plot_max_values(input_file):
    """
    Plot max frame value for each half second throughout the video.
    """
    clip = VideoFileClip(input_file)
    # Store the max value of frame seen every quarter second
    print("Starting analysis...")
    max_values = []
    max_values_per_6_frames = []
    import os
    import pickle
    
    # Define the cache file path
    cache_file = f"{input_file}_max_values.pkl"
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        print("Cached!")
        # Load the max values from the cache file
        with open(cache_file, 'rb') as f:
            max_values = pickle.load(f)
    else:
        print("Not cached!")
        # Calculate the max values and store them in the cache file
        for i, (t, frame) in enumerate(clip.iter_frames(with_times=True)):
            max_frame = np.max(frame, axis=(0, 1))
            max_values.append(max_frame)
        with open(cache_file, 'wb') as f:
            pickle.dump(max_values, f)
    
    # Calculate max_values_per_6_frames from every 6 frames in max_values
    for i in range(0, len(max_values), 6):
        frames_to_average = max_values[i:i+6]
        max_frame = np.max(frames_to_average, axis=0)
        max_values_per_6_frames.append(max_frame)
        
    print("Max every 6 frames", max_values_per_6_frames)
    plt.plot([x/4 for x in range(len(max_values_per_6_frames))], [np.max(val) for val in max_values_per_6_frames])
    plt.title('Max frame value per quarter second')
    plt.xlim(0, len(max_values_per_6_frames)/4)  # Set x-axis range to match the number of data points in seconds
    
    # Find all the time ranges where at least 15 consecutive frames have a max below 190
    dimmed_ranges = []
    count = 0
    start_time = 0
    for i, value in enumerate(max_values):
        if np.max(value) < 190:
            if count == 0:
                start_time = i / 24  # Convert frame index to time in seconds
            count += 1
        else:
            if count >= 15:
                dimmed_ranges.append((start_time, i / 24))  # Add the start and end time of the range
            count = 0
    if count >= 15:
        dimmed_ranges.append((start_time, len(max_values) / 24))  # Add the last range if it ends at the end of the video
    
    # Print characteristics of the values in each range
    for start, end in dimmed_ranges:
        range_values = max_values[int(start*24):int(end*24)]  # Get the values in the range
        print(f"Time range with at least 3 consecutive values below 180: {int(start/60)}:{start%60:.2f} - {int(end/60)}:{end%60:.2f} minutes")
        print(f"Average value: {np.mean([np.max(val) for val in range_values])}")
        print(f"Max value: {np.max([np.max(val) for val in range_values])}")
        print(f"Min value: {np.min([np.min(val) for val in range_values])}")
        print(f"Variance: {np.var([np.var(val) for val in range_values])}")
        if len(range_values) > 0:
            filtered_values = [x for x in range_values if isinstance(x, np.ndarray) and x.shape == (3,)]
            if len(filtered_values) > 0:
                print(f"Variance between channels: {np.var(filtered_values, axis=0)}")
        
        range_values = [np.max(frame, axis=(0, 1)) for frame in clip.iter_frames()][int(start*24):int(end*24)]  # Get the values in the range
        risk = calculate_epilepsy_risk(range_values)
        print(f"Epileptic risk: ", risk)
        
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