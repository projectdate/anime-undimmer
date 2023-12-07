import argparse
from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

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
    """
    Calculate the risk of epilepsy for a video.

    This function calculates the mean and standard deviation of the absolute sum of differences between consecutive frames.
    A high mean and standard deviation indicates a high risk of epilepsy.

    Parameters:
    frame_values_gen (generator): A generator that yields the pixel values for each frame in the video.

    Returns:
    mean, stddev: A tuple containing the mean and standard deviation of the absolute sum of differences between consecutive frames.
    """
    # Initialize variables
    prev_frame_values = next(frame_values_gen)
    abs_sum_diffs = []
    # Iterate over the generator
    for frame_values in frame_values_gen:
        # Calculate the difference between consecutive frames
        frame_diffs = frame_values.astype(int) - prev_frame_values.astype(int)        # Calculate the absolute sum of pixel differences for each frame difference
        abs_sum_diffs.append(np.mean(np.abs(frame_diffs)))
        # Update previous frame values
        prev_frame_values = frame_values

    frame_count = len(abs_sum_diffs)
    # Convert list to numpy array
    abs_sum_diffs = np.array(abs_sum_diffs)
    # Calculate the proportion of abs_sum_diffs that are more than 20
    flash_proportion = np.sum(abs_sum_diffs > 20) / frame_count
    # Calculate the mean of the absolute sum of differences
    risk_mean = np.mean(abs_sum_diffs)
    # Calculate the standard deviation of the absolute sum of differences
    risk_stddev = np.std(abs_sum_diffs)
    # Print the mean and standard deviation of the absolute sum of differences
    # print(f"Mean consecutive frames difference: {risk_mean}")
    # print(f"Standard deviation of consecutive frame differences: {risk_stddev}")
    # If the standard deviation is high, the video is more likely to cause epilepsy
    print(f"Epileptic risk: {risk_mean:.1f}, {risk_stddev:.1f}. Flashes: {flash_proportion:.1f} in {frame_count:.1f} frames")
    if risk_mean > 10:
        # and risk_stddev > 8:
        return True
    return False

def calculate_epilepsy_risk_v2(frame_values_gen):
    """
    Calculate the risk of epilepsy for a video based on luminescence calculations.
    
    Unfortunately, this isn't tuned very well and seems to fail.

    This function calculates the risk based on the following guidelines:
    - A potentially harmful flash occurs when there is a pair of opposing changes in luminance of 20cd/m2 or more.
    - This applies only when the screen luminance of the darker image is below 160cd/m2.
    - Irrespective of luminance, a transition to or from a saturated red is also potentially harmful.
    - A sequence of flashes is not permitted when both the following occur: 
        (a) the combined area of flashes occurring concurrently occupies more than 25% of the displayed screen area and 
        (b) the flash frequency is higher than 3Hz.
    - A sequence of flashing images lasting more than 5s might constitute a risk even when it complies with the guidelines above.
    - Rapidly changing image sequences are provocative if they result in areas of the screen that flash, in which case the same constraints apply as for flashes.

    - Based on the Ofcom guidelines specified here: https://arxiv.org/pdf/2108.09491.pdf and here: https://trace.umd.edu/peat/ and here: https://ieeexplore.ieee.org/document/7148104
    Parameters:
    frame_values_gen (generator): A generator that yields the pixel values for each frame in the video.

    Returns:
    risk: A boolean indicating whether the video poses a risk of epilepsy.
    """
    # Initialize variables
    prev_frame_values = next(frame_values_gen)
    flash_count = 0
    flash_duration = 0
    fps = 24
    risk = False

    # Initialize a variable to keep track of the number of frames
    frame_count = 0
    i = 0
    
    # Iterate over the generator
    for frame_values in frame_values_gen:
        # Calculate the difference between consecutive frames
        frame_diffs = frame_values.astype(int) - prev_frame_values.astype(int)
        # Check for harmful flash
        if np.mean(np.abs(frame_diffs)) > 20 and (np.mean(prev_frame_values) < 160 or np.mean(frame_values) < 160):
            flash_count += 1
            
        # Check for transition to or from saturated red
        red_pixels_current_frame = np.isclose(frame_values, [255, 0, 0], atol=50)
        if np.mean(red_pixels_current_frame) > 0.1:
            flash_count += 1
            
        # Update frame count
        frame_count += 1
        
        # Check for sequence of flashes per second, if less than 8 frames apart then mark risky
        if flash_count >= 3 and frame_count <= fps:
            print(i, " frames in, too many flashes detected")
            risk = True
            break
        elif flash_count >= 3:
            flash_count = 0
            frame_count = 0
        
        # Update previous frame values
        prev_frame_values = frame_values
        i += 1
        
    return risk

def frame_generator(clip, start, end):
    start_frame = int(start)
    end_frame = int(end)
    for i, frame in enumerate(clip.iter_frames()):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        yield frame

# Calculates mean of the max luminescence of each frame
def calculate_mean_without_outliers(values):
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    threshold_values = [x for x in values if ((q25 - 1.5*iqr) <= np.max(x) <= (q75 + 1.5*iqr))]
    return np.mean([np.max(val) for val in threshold_values])

def load_values(input_file, clip):
    """
    Load the max and avg values from the cache file if it exists, otherwise calculate them and store them in the cache file.
    """
    print("Starting analysis...")
    max_values = []
    avg_values = []
    
    # Define the cache file path
    cache_file = f"{input_file}_max_and_avg_values.pkl"
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        print("Cached!")
        # Load the max and avg values from the cache file
        with open(cache_file, 'rb') as f:
            max_values, avg_values = pickle.load(f)
    else:
        print("Not cached!")
        # Calculate the max and avg values and store them in the cache file
        for i, (t, frame) in enumerate(clip.iter_frames(with_times=True)):
            max_frame = np.max(frame, axis=(0, 1))
            avg_frame = np.mean(frame, axis=(0, 1))
            max_values.append(max_frame)
            avg_values.append(avg_frame)
        with open(cache_file, 'wb') as f:
            pickle.dump((max_values, avg_values), f)
    
    print("Loaded max and avg values!")
    return max_values, avg_values

def calculate_max_values_per_6_frames(max_values):
    """
    Calculate max_values_per_6_frames from every 6 frames in max_values.
    """
    max_values_per_6_frames = []
    for i in range(0, len(max_values), 6):
        frames_to_average = max_values[i:i+6]
        max_frame = np.max(frames_to_average, axis=0)
        max_values_per_6_frames.append(max_frame)
    return max_values_per_6_frames

def plot_max_values(max_values_per_6_frames):
    """
    Plot the max_values_per_6_frames and show the plot.
    """
    plt.plot([x/4 for x in range(len(max_values_per_6_frames))], [np.max(val) for val in max_values_per_6_frames])
    plt.title('Max frame value per quarter second')
    # plt.xlim(0, len(max_values_per_6_frames)/4)  # Set x-axis range to match the number of data points in seconds
    plt.show()

def find_dark_and_dimmed_ranges(max_values, threshold):
    """
    Find all the time ranges where at least 15 consecutive frames have a max below threshold.
    """
    dark_and_dimmed_ranges = []
    count = 0
    start_time = 0
    for i, value in enumerate(max_values):
        if np.max(value) < threshold:
            if count == 0:
                start_time = i  # Convert frame index to time in seconds
            count += 1
        else:
            if count >= 15:
                dark_and_dimmed_ranges.append((start_time, i))  # Add the start and end time of the range
            count = 0
    if count >= 15:
        dark_and_dimmed_ranges.append((start_time, len(max_values)))  # Add the last range if it ends at the end of the video
    return dark_and_dimmed_ranges

def print_range_characteristics(clip, max_values, dark_and_dimmed_ranges):
    """
    Print characteristics of the values in each range.
    """
    dimmed_ranges = []
    for start, end in dark_and_dimmed_ranges:
        range_values = max_values[int(start):int(end)]  # Get the values in the range
        print(f"Possible dark or dimmed time range: {int((start / 24)/60)}:{(start / 24)%60:.2f} - {int((end / 24)/60)}:{(end / 24)%60:.2f} minutes")
        avg_value = np.mean([np.max(val) for val in range_values])
        max_value = np.max([np.max(val) for val in range_values])
        mean_value_no_outliers = calculate_mean_without_outliers(range_values)
        min_value = np.min([np.min(val) for val in range_values])
        variance = np.var([np.var(val) for val in range_values])
        print(f"Average value: {avg_value:.2f}, Max value: {max_value}, Min value: {min_value}, Mean without Outliers: {mean_value_no_outliers}, Variance: {variance:.2f}")
        if len(range_values) > 0:
            filtered_values = [x for x in range_values if isinstance(x, np.ndarray) and x.shape == (3,)]
            if len(filtered_values) > 0:
                print(f"Variance between channels: {[round(var, 2) for var in np.var(filtered_values, axis=0)]}")
        
        # Get the exact frame values in the range using a generator to avoid creating a large temporary list
        exact_frame_values = frame_generator(clip, start, end)
        is_epileptic = calculate_epilepsy_risk(exact_frame_values)
        if is_epileptic:
            print(f"Likely dimmed scene! Undimming range:  ({start}, {end}, {256 / avg_value:.2f})")
            # TODO: Instead of putting 256, put the neighboring scene maxes
            dimmed_ranges.append((start, end, 256 / avg_value))
        print("")
        
    return dimmed_ranges

def get_dimmed_scenes(input_file, show_plot, threshold):
    """
    Return (or plot) scene ranges with dim factors throughout the video.
    This function takes in an input video file and a boolean value for whether to show the plot or not.
    If show_plot, it only calculates the maximum frame value for each half second throughout the video and plots it.
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
    max_values, avg_values = load_values(input_file, clip)
    max_values_per_6_frames = calculate_max_values_per_6_frames(max_values)
    
    if show_plot:
        plot_max_values(max_values_per_6_frames)
        return []
    
    dark_and_dimmed_ranges = find_dark_and_dimmed_ranges(max_values, threshold)
    dimmed_ranges = print_range_characteristics(clip, max_values, dark_and_dimmed_ranges)
    return dimmed_ranges

def get_dim_factor(input_file, start, end):
    """
    Given a time range, return the dim factor.
    
    Parameters:
    input_file (str): The path to the input video file.
    start (int): The start frame number of the time range.
    end (int): The end frame number of the time range.

    Returns:
    float: The dim factor for the given time range.
    """
    clip = VideoFileClip(input_file)
    max_values, avg_values = load_values(input_file, clip)
    range_values = max_values[int(start):int(end)]  # Get the values in the range
    avg_value = np.mean([np.max(val) for val in range_values])
    mean_value_no_outliers = calculate_mean_without_outliers(range_values)
    dim_factor = 256 / avg_value
    print("Dim factor autocalculated: ", dim_factor)
    return dim_factor


def get_all_dimmed_scenes(input_file, show_plot):
    """
    Return (or plot) scene ranges with dim factors throughout the video for a variety of dim thresholds.
    This function takes in an input video file and a boolean value for whether to show the plot or not.
    If show_plot, it only calculates the maximum frame value for each half second throughout the video and plots it.
    If show_plot is True, it will display the plot and return an empty list.
    If show_plot is False, it will not display the plot and will return a list of time ranges where at least 15 consecutive frames have a max below 190.
    The returned time ranges represent the dimmed scenes in the video, ordered by most dim first.
    
    Parameters:
    input_file (str): The path to the input video file.
    show_plot (bool): Whether to display the plot or not.
    
    Returns:
    list: A list of time ranges (start frame, end frame, factor) representing the dimmed scenes in the video and how much to undim them by, if show_plot is False. An empty list if show_plot is True.
    """
    thresholds = [150, 190, 230]
    dimmed_scenes = []
    thresholds.sort() # This is neccessary due to the way we apply most aggressive filters first
    for threshold in thresholds:
        dimmed_scenes.extend(get_dimmed_scenes(input_file, show_plot, threshold))
    return dimmed_scenes

def main():
    parser = argparse.ArgumentParser(description='Multiply color values in a video by a factor.')
    parser.add_argument('input_file', type=str, help='Path to the input video file')
    parser.add_argument('output_file', type=str, help='Path to the output video file')
    parser.add_argument('--only_plot', action='store_true', help='Only plot max frame value for each quarter second')
    parser.add_argument('--custom_scene', nargs=3, metavar=('start', 'end', 'factor'), help='Define a custom dimmed scene with start time, end time (in minutes:seconds or seconds), and optional dim factor (if 0, we will auto-calculate)')

    args = parser.parse_args()

    def time_to_frame(time_str):
        if ':' in time_str:
            minutes, seconds = map(float, time_str.split(':'))
        else:
            minutes = 0
            seconds = float(time_str)
        return int((minutes * 60 + seconds) * 24)  # assuming 24 frames per second

    if(args.only_plot):
        get_dimmed_scenes(args.input_file, args.only_plot, 0)
    else:
        if args.custom_scene:
            start, end, factor = args.custom_scene
            start = time_to_frame(start)
            end = time_to_frame(end)
            factor = float(factor)
            factor = factor if factor > 0.0 else get_dim_factor(args.input_file, start, end)
            dimmed_scenes = [(start, end, factor)]
        else:
            dimmed_scenes = get_all_dimmed_scenes(args.input_file, args.only_plot)
        print("Dimmed scenes (start frame, stop frame, dim factor): ", dimmed_scenes)
        process_video(args.input_file, args.output_file, dimmed_scenes)

if __name__ == "__main__":
    main()