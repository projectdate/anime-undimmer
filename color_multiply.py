import argparse
from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt

def multiply_colors(frame, dimmed_scenes, current_frame):
    """
    Multiply each color in the frame by a given factor.
    Clipping is performed to ensure pixel values stay within valid range.
    Only multiply colors in the dimmed scenes range.
    """
    for start, end, factor in dimmed_scenes:
        if start <= current_frame < end:
            return np.clip(frame * factor, 0, 255).astype('uint8')
    return frame

def process_video(input_file, output_file, dimmed_scenes):
    """
    Process the video, multiplying each frame's pixel values by the specified factor.
    Only multiply colors in the dimmed scenes range.
    """
    clip = VideoFileClip(input_file)
    clip = clip.fl(lambda gf, t: multiply_colors(gf(t), dimmed_scenes, int(t*clip.fps)))
    clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

def calculate_epilepsy_risk(frame_values_gen):
    # Initialize variables
    prev_frame_values = next(frame_values_gen)
    abs_sum_diffs = []
    # Iterate over the generator
    for frame_values in frame_values_gen:
        # Calculate the difference between consecutive frames
        frame_diffs = frame_values - prev_frame_values
        # Calculate the absolute sum of pixel differences for each frame difference
        abs_sum_diffs.append(np.mean(np.abs(frame_diffs)))
        # Update previous frame values
        prev_frame_values = frame_values

    # Convert list to numpy array
    abs_sum_diffs = np.array(abs_sum_diffs)
    # Calculate the mean of the absolute sum of differences
    mean_abs_sum_diff = np.mean(abs_sum_diffs)
    # Calculate the standard deviation of the absolute sum of differences
    std_abs_sum_diff = np.std(abs_sum_diffs)
    # Print the mean and standard deviation of the absolute sum of differences
    print(f"Mean consecutive frames difference: {mean_abs_sum_diff}")
    print(f"Standard deviation of consecutive frame differences: {std_abs_sum_diff}")
    # If the standard deviation is high, the video is more likely to cause epilepsy
    return mean_abs_sum_diff, std_abs_sum_diff

def frame_generator(clip, start, end):
    start_frame = int(start)
    end_frame = int(end)
    for i, frame in enumerate(clip.iter_frames()):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        yield frame


def get_dimmed_scenes(input_file, show_plot):
    """
    Plot max frame value for each half second throughout the video.
    This function takes in an input video file and a boolean value for whether to show the plot or not.
    It calculates the maximum frame value for each half second throughout the video and plots it.
    If show_plot is True, it will display the plot and return an empty list.
    If show_plot is False, it will not display the plot and will return a list of time ranges where at least 15 consecutive frames have a max below 190.
    The returned time ranges represent the dimmed scenes in the video.
    
    Parameters:
    input_file (str): The path to the input video file.
    show_plot (bool): Whether to display the plot or not.
    
    Returns:
    list: A list of time ranges (start frame, end frame, factor) representing the dimmed scenes in the video and how much to undim them by, if show_plot is False. An empty list if show_plot is True.
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
    
    print("Loaded max values!")
    
    # Calculate max_values_per_6_frames from every 6 frames in max_values
    for i in range(0, len(max_values), 6):
        frames_to_average = max_values[i:i+6]
        max_frame = np.max(frames_to_average, axis=0)
        max_values_per_6_frames.append(max_frame)
        
    print("Max every 6 frames", max_values_per_6_frames)
    plt.plot([x/4 for x in range(len(max_values_per_6_frames))], [np.max(val) for val in max_values_per_6_frames])
    plt.title('Max frame value per quarter second')
    plt.xlim(0, len(max_values_per_6_frames)/4)  # Set x-axis range to match the number of data points in seconds
    if show_plot:
        plt.show()
        return []
    
    # Find all the time ranges where at least 15 consecutive frames have a max below 190
    dark_and_dimmed_ranges = []
    count = 0
    start_time = 0
    for i, value in enumerate(max_values):
        if np.max(value) < 190:
            if count == 0:
                start_time = i  # Convert frame index to time in seconds
            count += 1
        else:
            if count >= 15:
                dark_and_dimmed_ranges.append((start_time, i))  # Add the start and end time of the range
            count = 0
    if count >= 15:
        dark_and_dimmed_ranges.append((start_time, len(max_values)))  # Add the last range if it ends at the end of the video
    
    # Print characteristics of the values in each range
    # Time range with at least 15 consecutive values below 193
    dimmed_ranges = []
    for start, end in dark_and_dimmed_ranges:
        range_values = max_values[int(start):int(end)]  # Get the values in the range
        print(f"Possible dark or dimmed time range: {int((start / 24)/60)}:{(start / 24)%60:.2f} - {int((end / 24)/60)}:{(end / 24)%60:.2f} minutes")
        avg_value = np.mean([np.max(val) for val in range_values])
        max_value = np.max([np.max(val) for val in range_values])
        min_value = np.min([np.min(val) for val in range_values])
        variance = np.var([np.var(val) for val in range_values])
        print(f"Average value: {avg_value}, Max value: {max_value}, Min value: {min_value}, Variance: {variance}")
        if len(range_values) > 0:
            filtered_values = [x for x in range_values if isinstance(x, np.ndarray) and x.shape == (3,)]
            if len(filtered_values) > 0:
                print(f"Variance between channels: {np.var(filtered_values, axis=0)}")
        
        # Get the exact frame values in the range using a generator to avoid creating a large temporary list
        exact_frame_values = frame_generator(clip, start, end)
        risk_mean, risk_stddev = calculate_epilepsy_risk(exact_frame_values)
        print(f"Epileptic risk: {risk_mean:.1f}, {risk_stddev:.1f}")
        if risk_mean > 75 and risk_stddev > 7:
            print("Likely dimmed scene! Undimming range: ", (start, end, 256 / max_value))
            # TODO: Instead of putting 256, put the neighboring scene maxes
            dimmed_ranges.append((start, end, 256 / max_value))
        print("")
        
    return dimmed_ranges

def main():
    parser = argparse.ArgumentParser(description='Multiply color values in a video by a factor.')
    parser.add_argument('input_file', type=str, help='Path to the input video file')
    parser.add_argument('output_file', type=str, help='Path to the output video file')
    # parser.add_argument('factor', type=float, help='Factor to multiply color values by')
    parser.add_argument('--only_plot', action='store_true', help='Only plot max frame value for each quarter second')

    args = parser.parse_args()

    dimmed_scenes = get_dimmed_scenes(args.input_file, args.only_plot)
    print("Dimmed scenes (start, stop, dim factor): ", dimmed_scenes)
    if (not args.only_plot):
        process_video(args.input_file, args.output_file, dimmed_scenes)

if __name__ == "__main__":
    main()