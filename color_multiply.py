import argparse
import itertools
import math
import random
import shutil
import subprocess
from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import numpy.testing as npt
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
import concurrent.futures
import tqdm
import psutil
import modal

stub = modal.Stub(
    "parallel_clip",
    image=modal.Image.debian_slim().pip_install("argparse", "moviepy", "numpy", "matplotlib", "tqdm", "joblib", "psutil", "duckdb"),
)

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
    clip: VideoFileClip = VideoFileClip(input_file)
    clip = clip.fl(lambda gf, t: multiply_colors(gf(t), dimmed_scenes, int(t*clip.fps)))
    clip.write_videofile(output_file, codec='libx264', audio_codec='aac', threads=4)

# A specialized chunk function that re-emits the last value for diff calculation
def chunked(generator, chunk_size):
    chunk = []
    for item in generator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = [chunk[-1]]  # start the next chunk with the last frame of the current chunk
    if chunk:
        yield chunk
        
def calculate_epilepsy_risk(frame_values_gen, frame_values_gen_2, range_max_values, range_avg_values, range_diff_values, dim_multiplier):
    """
    Calculate the risk of epilepsy for a video.

    This function calculates the mean and standard deviation of the absolute sum of differences between consecutive frames.
    A high mean and standard deviation indicates a high risk of epilepsy.

    Parameters:
    frame_values_gen (generator): A generator that yields the pixel values for each frame in the video.

    Returns:
    mean, stddev: A tuple containing the mean and standard deviation of the absolute sum of differences between consecutive frames.
    """
    # Convert lists to numpy arrays
    abs_sum_diffs = range_diff_values
    abs_luminescance = np.mean(range_avg_values, axis=1)
    
    frame_count = len(abs_sum_diffs)
    # Convert list to numpy array
    abs_sum_diffs = np.array(abs_sum_diffs)
    # Calculate the proportion of abs_sum_diffs that are more than 20
    # Technically, since the entire range has been dimmed by some factor, this 20 should decrease by the same factor since this check is pre-dimming
    flash_count = np.sum(abs_sum_diffs > 20)
    # This corrects the above technicality. Holding off on making it the default though.
    flash_count_corrected = np.sum(abs_sum_diffs > (20 / dim_multiplier))
    
    # Count the number of flashes less than 9 frames apart
    # Create a boolean array where True represents a flash
    is_flash = abs_sum_diffs > (20 / dim_multiplier)
    # Find the indices where a flash occurs
    flash_indices = np.where(is_flash)[0]
    # Calculate the differences between consecutive flash indices
    flash_diffs = np.diff(flash_indices)
    # Count the number of flashes that are less than 9 frames apart
    close_flash_count = np.sum(flash_diffs < 9)
    
    # Calculate the number of flashes where the dimmed scene is below 160
    # Doesn't work since doesn't consider saturated red: This applies only when the screen luminance of the darker image is below 160 cd/m2. Irrespective of luminance, a transition to or from a saturated red is also potentially harmful. 
    flash_count_below_160 = np.sum((abs_sum_diffs > 20) & ((np.array(abs_luminescance[:-1]) < 160) | (np.array(abs_luminescance[1:]) < 160)))
    # Corrects the same predim/postdim discrepancy
    dimmed_160 = 160 / dim_multiplier
    flash_count_below_160_corrected = np.sum((abs_sum_diffs > (20 / dim_multiplier)) & ((np.array(abs_luminescance[:-1]) < dimmed_160) | (np.array(abs_luminescance[1:]) < dimmed_160)))
    # Calculate the mean of the absolute sum of differences
    risk_mean = np.mean(abs_sum_diffs)
    # Calculate the standard deviation of the absolute sum of differences
    risk_stddev = np.std(abs_sum_diffs)
    # Print the mean and standard deviation of the absolute sum of differences
    # print(f"Mean consecutive frames difference: {risk_mean}")
    # print(f"Standard deviation of consecutive frame differences: {risk_stddev}")
    # If the standard deviation is high, the video is more likely to cause epilepsy
    print(f"Epileptic risk: {risk_mean:.1f}, {risk_stddev:.1f}. \
Flashes: {flash_count / frame_count:.2f}, {flash_count} in {frame_count} frames, \
Flashes with a <160: {flash_count_below_160 / frame_count:.2f}, {flash_count_below_160} in {frame_count} frames, \
Predim flashes: {flash_count_corrected / frame_count:.2f}, {flash_count_corrected} in {frame_count} frames, \
Predim flashes with a predim <160: {flash_count_below_160_corrected / frame_count:.2f}, {flash_count_below_160_corrected} in {frame_count} frames, \
Predim flashes less than 9 frames apart: {close_flash_count / frame_count:.2f}, {close_flash_count} in {frame_count} frames")
    
    # We should really check if any 2 flashes are less than 9 frames apart:
    #  For clarification, successive flashes for which the leading edges are separated by nine frames or more are acceptable in a 50 Hz environment, or separated by ten frames or more are acceptable in a 60 Hz environment, irrespective of their brightness or screen area. 
    
    # This isn't strictly the criteria -- flash count should technically exceed 3 in any 1 second segment, not a segment of any length
    # Calculating that with a generator requires some fancy offset and duplication logic (i.e. each 1 gets projected to a 1 for the next 24 frames, then we add them all and see if there's a 3 anywhere)
    # Till then, this has worked well enough
    
    if risk_mean > 10 or flash_count >= 3:
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

# Takes the whole clip and returns a generator that yields frames from start to end
def frame_generator(clip, start, end):
    start_frame = int(start)
    end_frame = int(end)
    for i, frame in enumerate(clip.iter_frames()):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        yield frame

# Calculates mean of the luminescence of each frame, removing outliers
def calculate_mean_without_outliers(values):
    """
    This function calculates the mean of the luminescence of each frame, removing outliers.

    Parameters:
    values (np.array): A numpy array of luminescence values of each frame. Each frame can be a full RGB frame or a single value.

    Returns:
    float: The mean of the luminescence values after removing outliers.
    """
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    threshold_values = [x for x in values if ((q25 - 1.5*iqr) <= np.max(x) <= (q75 + 1.5*iqr))]
    return np.mean([np.max(val) for val in threshold_values])

# Calculates mean of the luminescence of each frame, removing outliers
def calculate_max_without_outliers_single_frame(frame):
    """
    This function calculates the mean of the luminescence of each frame, removing outliers.

    Parameters:
    frame (np.array): A numpy array representing a single frame. The array is 3D with dimensions (height, width, RGB).

    Returns:
    float: The mean of the luminescence values after removing outliers.
    """
    q75, q25 = np.percentile(frame, [75, 25])
    iqr = q75 - q25
    mask = np.logical_and((q25 - 1.5*iqr) <= frame, frame <= (q75 + 1.5*iqr))
    # Ensure the mask is broadcasted across the RGB channels
    threshold_frames = frame * mask  # Add a new axis for broadcasting
    # Now, calculate the max without outliers for each channel separately
    max_no_outliers_frame = np.max(threshold_frames, axis=(0, 1))  # Max across height and width, but keep the RGB channels separate
    return max_no_outliers_frame

@stub.function(cpu=14)
def process_frame(frame, prev_frame):
    """
    Process a frame and calculate the maximum, average and difference values. Time and print it.

    Parameters:
    frame (np.array): The current frame to be processed. It's a 3D array (height, width, RGB).
    prev_frame (np.array): The previous frame for difference calculation. It's a 3D array (height, width, RGB).

    Returns:
    tuple: A tuple containing maximum (1D array of RGB values), average (1D array of RGB values) and difference values (single float value) of the frame.
    """
    import time
    
    start_time_max_frame = time.time()
    max_frame = np.max(frame, axis=(0, 1))
    end_time_max_frame = time.time()
    
    start_time_avg_frame = time.time()
    avg_frame = np.mean(frame, axis=(0, 1))
    end_time_avg_frame = time.time()
    
    start_time_max_no_outliers_frame = time.time()
    max_no_outliers_frame = calculate_max_without_outliers_single_frame(frame)
    end_time_max_no_outliers_frame = time.time()
    
    if prev_frame is None: # To align the size of the arrays
        prev_frame = frame
    
    start_time_diff_frame = time.time()
    diff_frame = np.subtract(frame.astype(int), prev_frame.astype(int))
    end_time_diff_frame = time.time()
    
    start_time_diff_value = time.time()
    diff_value = np.mean(np.abs(diff_frame))
    end_time_diff_value = time.time()
    
    # Time taken for max_frame: 0.009861946105957031 seconds, avg_frame: 0.009123086929321289 seconds, max_no_outliers_frame: 0.026125669479370117 seconds, diff_frame: 0.00890803337097168 seconds,  diff_value: 0.003439188003540039 seconds
    # Time taken for max_frame: 0.008923053741455078 seconds, avg_frame: 0.010212898254394531 seconds, max_no_outliers_frame: 0.026179075241088867 seconds, diff_frame: 0.009123086929321289 seconds, diff_value: 0.004380941390991211 seconds
    # Time taken for max_frame: 0.00949406623840332 seconds,  avg_frame: 0.00931096076965332 seconds,  max_no_outliers_frame: 0.025065898895263672 seconds, diff_frame: 0.009306192398071289 seconds, diff_value: 0.0051729679107666016 seconds
    # Time taken for max_frame: 0.008934974670410156 seconds, avg_frame: 0.009266853332519531 seconds, max_no_outliers_frame: 0.01978325843811035 seconds,  diff_frame: 0.011101961135864258 seconds, diff_value: 0.004160881042480469 seconds
    # Time taken for max_frame: 0.008736133575439453 seconds, avg_frame: 0.009737014770507812 seconds, max_no_outliers_frame: 0.020854949951171875 seconds, diff_frame: 0.015026092529296875 seconds, diff_value: 0.004965066909790039 seconds
    # Time taken for max_frame: 0.008708000183105469 seconds, avg_frame: 0.008868932723999023 seconds, max_no_outliers_frame: 0.022265911102294922 seconds, diff_frame: 0.009536981582641602 seconds, diff_value: 0.003737926483154297 seconds
    # print(f"Time taken for max_frame: {end_time_max_frame - start_time_max_frame} seconds, avg_frame: {end_time_avg_frame - start_time_avg_frame} seconds, max_no_outliers_frame: {end_time_max_no_outliers_frame - start_time_max_no_outliers_frame} seconds, diff_frame: {end_time_diff_frame - start_time_diff_frame} seconds, diff_value: {end_time_diff_value - start_time_diff_value} seconds")
    
    return max_frame, max_no_outliers_frame, avg_frame, diff_value

@stub.function(cpu=14)
def process_frames_sequentially(frames_with_info, prev_frame):
    results = []
    for i_frame in frames_with_info:
        i, (t, frame) = i_frame
        result = process_frame(frame, prev_frame)
        results.append(result)
        prev_frame = frame  # Update prev_frame for the next iteration
    return results

def process_frame_parallel(frame, prev_frame):
    """
    Process a frame and calculate the maximum, average and difference values.
    This function works fine but only leads to about a 10% speedup.

    Parameters:
    frame (np.array): The current frame to be processed. It's a 3D array (height, width, RGB).
    prev_frame (np.array): The previous frame for difference calculation. It's a 3D array (height, width, RGB).

    Returns:
    tuple: A tuple containing maximum (1D array of RGB values), average (1D array of RGB values) and difference values (single float value) of the frame.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        max_frame = executor.submit(np.max, frame, axis=(0, 1)).result()
        avg_frame = executor.submit(np.mean, frame, axis=(0, 1)).result()
        max_no_outliers_frame = executor.submit(calculate_max_without_outliers_single_frame, frame).result()
        if prev_frame is None: # To align the size of the arrays
            prev_frame = frame
        diff_frame = executor.submit(np.subtract, frame.astype(int), prev_frame.astype(int)).result()
        diff_value = executor.submit(np.mean, np.abs(diff_frame)).result()
    return max_frame, max_no_outliers_frame, avg_frame, diff_value

def process_clip(clip):
    """
    Process a clip and calculate the maximum, average and difference values for each frame.

    Parameters:
    clip (VideoFileClip): The video clip to be processed. It's an object of class VideoFileClip.

    Returns:
    tuple: A tuple containing lists of maximum (1D array of RGB values), maximum without outliers (1D array of RGB values), average (1D array of RGB values) and difference values (1D array of single float values) for each frame in the clip.
    """
    start_time = time.time()
    max_values = []
    avg_values = []
    diff_values = []
    max_no_outliers_values = []
    prev_frame = clip.get_frame(0)
    for i, (t, frame) in tqdm.tqdm(enumerate(clip.iter_frames(with_times=True)), total=clip.fps*clip.duration, dynamic_ncols=True, desc="Processing frames"):
        if prev_frame is None: # To align the size of the arrays
            prev_frame = frame
        frame = np.array(frame)
        max_frame, max_no_outliers_frame, avg_frame, diff_frame = process_frame(frame, prev_frame)
        prev_frame = frame
        max_values.append(max_frame)
        max_no_outliers_values.append(max_no_outliers_frame)
        avg_values.append(avg_frame)
        diff_values.append(diff_frame)
    end_time = time.time()
    print(f"Time taken to calculate params (slow): {(end_time - start_time) * 1000:.2f} ms")
    return max_values, max_no_outliers_values, avg_values, diff_values

@stub.function(cpu=14)
def process_clip_parallel_modal_old(clip):
    """
    Process a clip in parallel and calculate the maximum, average and difference values for each frame.
    This is a parallel version of process_clip, but it OOMs on large clips.
    
    Parameters:
    clip (VideoFileClip): The video clip to be processed. It's an object of class VideoFileClip.

    Returns:
    tuple: A tuple containing lists of maximum (1D array of RGB values), maximum without outliers (1D array of RGB values), average (1D array of RGB values) and difference values (1D array of single float values) for each frame in the clip.
    """
    # # Calculate the max, avg and diff values and store them in the cache file in parallel
    # I think this can work up to some number of frames or something? Or else it OOMs   
    start_time = time.time()
    frames = list(clip.iter_frames(with_times=True))
    cores = psutil.cpu_count(logical=False)
    print(f"Using max memory on {20} cores, {cores} available...")
    results = Parallel(20)(delayed(process_frame)(frame, frames[i-1][1] if i > 0 else frame) 
                                for i, (t, frame) in tqdm.tqdm(enumerate(frames), total=clip.fps*clip.duration, dynamic_ncols=True))
    max_values, max_no_outliers_values, avg_values, diff_values = zip(*results)
    end_time = time.time()
    print(f"Time taken to calculate params (parallel): {(end_time - start_time) * 1000:.2f} ms")
    return max_values, max_no_outliers_values, avg_values, diff_values

@stub.local_entrypoint()
def process_clip_parallel_modal_new(clip):
    """
    Process a clip in parallel and calculate the maximum, average and difference values for each frame.
    This is a parallel version of process_clip, but it OOMs on large clips.
    
    Parameters:
    clip (VideoFileClip): The video clip to be processed. It's an object of class VideoFileClip.

    Returns:
    tuple: A tuple containing lists of maximum (1D array of RGB values), maximum without outliers (1D array of RGB values), average (1D array of RGB values) and difference values (1D array of single float values) for each frame in the clip.
    """
    # # Calculate the max, avg and diff values and store them in the cache file in parallel
    # I think this can work up to some number of frames or something? Or else it OOMs   
    start_time = time.time()
    frames = list(clip.iter_frames(with_times=True))
    cores = psutil.cpu_count(logical=False)
    batch_size = 20
    print(f"Using max memory on {batch_size} cores, {cores} available...")
    results = Parallel(batch_size)(delayed(process_frame)(frame, frames[i-1][1] if i > 0 else frame) 
                                for i, (t, frame) in tqdm.tqdm(enumerate(frames), total=clip.fps*clip.duration, dynamic_ncols=True))
    max_values, max_no_outliers_values, avg_values, diff_values = zip(*results)
    end_time = time.time()
    print(f"Time taken to calculate params (parallel): {(end_time - start_time) * 1000:.2f} ms")
    return max_values, max_no_outliers_values, avg_values, diff_values

# joblib.externals.loky.process_executor.BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.
def process_clip_parallel(clip):
    """
    Process a clip in parallel and calculate the maximum, average and difference values for each frame.
    
    Parameters:
    clip (VideoFileClip): The video clip to be processed. It's an object of class VideoFileClip.

    Returns:
    tuple: A tuple containing lists of maximum (1D array of RGB values), maximum without outliers (1D array of RGB values), average (1D array of RGB values) and difference values (1D array of single float values) for each frame in the clip.
    """
    cores = psutil.cpu_count(logical=False)
    batch_size = 8
    print(f"Using {batch_size} batch size, {cores} cores available...")

    # Create a Parallel object with 8 jobs
    with Parallel(n_jobs=batch_size) as parallel:
        results = []
        frame_generator = clip.iter_frames(with_times=True)
        prev_frame = None
        # Get the total number of frames for the progress bar
        total_frames = int(clip.fps * clip.duration)

        # Create a progress bar
        pbar = tqdm.tqdm(total=total_frames, desc="Processing frames")

        while True:
            # Create a batch of jobs
            batch = list(itertools.islice(frame_generator, batch_size))
            if not batch:
                break
            # Process the batch of jobs in parallel
            batch_jobs = [(frame, prev_frame if i > 0 else frame) for i, (t, frame) in enumerate(batch)]
            # Compare the outputs of process_frame_parallel and process_frame on one frame
            batch_results = Parallel(n_jobs=batch_size)(delayed(process_frame_parallel)(*job) for job in batch_jobs)

            # Append the results to the results list
            results.extend(batch_results)
            # Update the previous frame
            prev_frame = batch[-1][1]
            # Update the progress bar
            pbar.update(len(batch))

    # Close the progress bar
    pbar.close()
    max_values, max_no_outliers_values, avg_values, diff_values = zip(*results)
    return max_values, max_no_outliers_values, avg_values, diff_values

def load_values(input_file, clip):
    """
    Load the max, avg values and diff between each pair of consecutive scenes from the cache file if it exists, otherwise calculate them and store them in the cache file.
    """
    print("Starting analysis...")
    max_values = []
    avg_values = []
    diff_values = []
    
    # Define the cache file path
    cache_file = f"{input_file}_max_avg_and_diff_values.pkl"
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        print("Cached!")
        # Load the max, avg and diff values from the cache file
        with open(cache_file, 'rb') as f:
            processed_values = pickle.load(f)
            if len(processed_values) == 3:
                max_values, avg_values, diff_values = processed_values
                max_no_outliers_values = max_values
            else:
                max_values, max_no_outliers_values, avg_values, diff_values = processed_values
            # max_values, max_no_outliers_values, avg_values, diff_values = pickle.load(f)  
    else:
        print("Not cached! Calculating params now...")
        max_values, max_no_outliers_values, avg_values, diff_values = process_clip_parallel(clip)
        print("Done calculating! Caching...")
        with open(cache_file, 'wb') as f:
            pickle.dump((max_values, max_no_outliers_values, avg_values, diff_values), f)
        print(f"Cached! Delete {cache_file} to clear it.")
    
    print("Loaded max, avg and diff values!")
    return max_values, max_no_outliers_values, avg_values, diff_values

def calculate_fn_per_frame_group(max_values, fn = np.max, frames = 6):
    """
    Calculate fn on every 6 frames in max_values.
    """
    fn_over_frame_group = []
    for i in range(0, len(max_values), frames):
        frames_to_average = max_values[i:i+6]
        max_frame = fn(frames_to_average, axis=0)
        fn_over_frame_group.append(max_frame)
    return fn_over_frame_group

def plot_values(max_values_per_n_frames, avg_values_per_n_frames, max_no_outliers_values_per_n_frames, group_size = 6):
    """
    Plot the max, max (no outliers), and average values (over each 6 frames, set by group_size) and show the plot.
    """
    plt.plot([x * group_size / 24 for x in range(len(max_values_per_n_frames))], [np.max(val) for val in max_values_per_n_frames], label='Max')
    plt.plot([x * group_size / 24 for x in range(len(max_no_outliers_values_per_n_frames))], [np.max(val) for val in max_no_outliers_values_per_n_frames], label='Max (no outliers)')
    plt.plot([x * group_size / 24 for x in range(len(avg_values_per_n_frames))], [np.mean(val) for val in avg_values_per_n_frames], label='Avg')
    plt.title('Max, max no outliers, and avg frame value per quarter second')
    plt.legend()
    # plt.xlim(0, len(max_values_per_n_frames)/4)  # Set x-axis range to match the number of data points in seconds
    plt.show()

def find_dark_and_dimmed_ranges(max_values, threshold):
    """
    Find all the time ranges where at least 15 consecutive frames have a max below threshold.
    Iterative and not numpy optimized so is pretty slow.
    """
    start_timer = time.time()
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
    end_timer = time.time()
    print(f"Time taken by find_dark_and_dimmed_ranges (iterative): {(end_timer - start_timer) * 1000} milliseconds")
    return dark_and_dimmed_ranges


def find_dark_and_dimmed_ranges_fast(max_values, threshold):
    """
    Find all the time ranges where at least 15 consecutive frames have a max below threshold.
    Numpy optimized so should be faster.
    """
    start_time = time.time()
    below_threshold = np.max(max_values, axis=1) < threshold
    diff = np.diff(below_threshold.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    # It's possible that there's an end before the first start
    if len(ends) > len(starts):
        starts = np.insert(starts, 0, 0)
        
    ranges = np.c_[starts, ends]
    ranges = ranges[ranges[:, 1] - ranges[:, 0] >= 15]
    end_time = time.time()
    print(f"Time taken by find_dark_and_dimmed_ranges (numpy optimized): {(end_time - start_time) * 1000} milliseconds")
    return ranges

def filter_and_print_range_characteristics(clip, max_values, avg_values, diff_values, dark_and_dimmed_ranges):
    """
    Print characteristics of the values in each range.
    """
    dimmed_ranges = []
    for start, end in dark_and_dimmed_ranges:
        range_max_values = max_values[int(start):int(end)]  # Get the values in the range
        range_avg_values = avg_values[int(start):int(end)]  # Get the values in the range
        range_diff_values = diff_values[int(start+1):int(end)]  # Get the values in the range
        result = get_and_print_single_range_characteristics(clip, start, end, range_max_values, range_avg_values, range_diff_values)
        if result:
            dimmed_ranges.append(result)
        print("") # Newline
        
    return dimmed_ranges

def get_and_print_single_range_characteristics(clip, start, end, range_max_values, range_avg_values, range_diff_values):
    print(f"Possible dark or dimmed time range: {frame_to_time(start)} - {frame_to_time(end)} minutes")
    avg_value = np.mean([np.max(val) for val in range_max_values])
    max_value = np.max([np.max(val) for val in range_max_values])
    mean_value_no_outliers = calculate_mean_without_outliers(range_max_values)
    min_value = np.min([np.min(val) for val in range_max_values])
    variance = np.var([np.var(val) for val in range_max_values])
    print(f"Average value: {avg_value:.2f}, Max value: {max_value}, Min value: {min_value}, Mean without Outliers: {mean_value_no_outliers}, Variance: {variance:.2f}")
    if len(range_max_values) > 0:
        filtered_values = [x for x in range_max_values if isinstance(x, np.ndarray) and x.shape == (3,)]
        if len(filtered_values) > 0:
            print(f"Variance between channels: {[round(var, 2) for var in np.var(filtered_values, axis=0)]}")
    
    # Get the exact frame values in the range using a generator to avoid creating a large temporary list
    exact_frame_values = frame_generator(clip, start, end)
    exact_frame_values_2 = frame_generator(clip, start, end)
    is_epileptic = calculate_epilepsy_risk(exact_frame_values, exact_frame_values_2, range_max_values, range_avg_values, range_diff_values, 256 / avg_value)
    if is_epileptic:
        print(f"Likely dimmed scene! Undimming range:  ({start}, {end}, {256 / avg_value:.2f})")
        # TODO: Instead of putting 256, put the neighboring scene maxes
        return (start, end, 256 / avg_value)
    else:
        print(f"Likely NOT dimmed scene, just dark! If it was, dim range:  ({start}, {end}, {256 / avg_value:.2f})")
        return None
        

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
    max_values, max_no_outliers_values, avg_values, diff_values = load_values(input_file, clip)
    n_frames = 1
    max_values_per_n_frames = calculate_fn_per_frame_group(max_values, np.max, n_frames)
    max_no_outliers_values_values_per_n_frames = calculate_fn_per_frame_group(max_no_outliers_values, np.max, n_frames)
    avg_values_per_n_frames = calculate_fn_per_frame_group(avg_values, np.mean, n_frames)
    
    if show_plot:
        plot_values(max_values_per_n_frames, avg_values_per_n_frames, max_no_outliers_values_values_per_n_frames, n_frames)
        return []
    
    print(f"Shape of max_values: {len(max_values)} {len(max_values[0])}")
    # assert max_values.shape == max_no_outliers_values.shape, "Shapes of max_values and max_no_outliers_values do not match"
    # Switch to max_no_outliers_values later
    dark_and_dimmed_ranges = find_dark_and_dimmed_ranges(max_values, threshold)
    dark_and_dimmed_ranges_fast = find_dark_and_dimmed_ranges_fast(max_values, threshold)
    # TODO: If this line is never printed then replace one with the other
    print(len(dark_and_dimmed_ranges))
    print(len(dark_and_dimmed_ranges_fast))
    # assert (len(dark_and_dimmed_ranges) == 0 and len(dark_and_dimmed_ranges_fast) == 0) or np.all(dark_and_dimmed_ranges == dark_and_dimmed_ranges_fast), "Mismatch between fast and slow methods of finding dark and dimmed ranges"
    dimmed_ranges = filter_and_print_range_characteristics(clip, max_values, avg_values, diff_values, dark_and_dimmed_ranges)
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
    # TODO: Don't cache all of load_values, replace with more generator code
    clip = VideoFileClip(input_file)
    max_values, max_no_outliers_values, avg_values, diff_values = load_values(input_file, clip)
    range_values = max_values[int(start):int(end)]  # Get the values in the range
    avg_value = np.mean([np.max(val) for val in range_values])
    mean_value_no_outliers = calculate_mean_without_outliers(range_values)
    dim_factor = 256 / avg_value
    # Print the range characteristics to use as debugging
    # TODO: Replace by get_and_print_single_range_characteristics
    filter_and_print_range_characteristics(clip, max_values, avg_values, diff_values, [(int(start),int(end))])
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
        print(f"---------CALCULATING FOR DIM PERCENT >= {((256 - threshold) / 256):.2f}----------------")
        dimmed_scenes.extend(get_dimmed_scenes(input_file, show_plot, threshold))
    return dimmed_scenes

def time_to_frame(time_str):
    if ':' in time_str:
        minutes, seconds = map(float, time_str.split(':'))
    else:
        minutes = 0
        seconds = float(time_str)
    return int((minutes * 60 + seconds) * 24)  # assuming 24 frames per second
    
def frame_to_time(frame_num):
    """
    Convert frame number to timestamp in minute:second format.
    
    Parameters:
    frame_num (int): The frame number to convert to timestamp.
    
    Returns:
    str: The timestamp in minute:second format corresponding to the frame number.
    """
    minutes = frame_num // (24 * 60)
    seconds = (frame_num / 24) % 60
    return f"{minutes:02d}:{seconds:.2f}"

# @stub.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(description='Multiply color values in a video by a factor.')
    parser.add_argument('input_file', type=str, help='Path to the input video file')
    parser.add_argument('--out', type=str, nargs='?', default=None, help='Path to the output video file')
    parser.add_argument('--modal', action='store_true', help='Pass if should run the expensive parallel processing on modal in the cloud for speed')
    parser.add_argument('--only_plot', action='store_true', help='Only plot max frame value for each quarter second')
    parser.add_argument('--custom_scene', nargs=3, metavar=('start', 'end', 'factor (0 to auto-calculate)'), help='Define a custom dimmed scene with start time, end time (in minutes:seconds or seconds), and optional dim factor (if 0, we will auto-calculate)')

    args = parser.parse_args()
    args.output_file = args.out
    
    if args.output_file is not None and args.only_plot:
        parser.error("--out will be ignored when --only_plot is set, remove output file")

    if args.output_file and os.path.splitext(args.input_file)[1] != os.path.splitext(args.output_file)[1]:
        raise ValueError("Input and output files must have the same file extension.")
    
    if args.output_file is None and not args.only_plot:
        parser.print_help("--out must specificy an output file if --only_plot is not set! Will continue with tmp.mkv.")
        args.output_file = "tmp.mkv"
          
    if args.output_file and os.path.splitext(args.output_file)[1] != '.mkv':
        raise ValueError("Output file must have .mkv extension.")
        
    if(args.modal):
        with stub.run():
            process_input(args)
    else:
        process_input(args)
    
def plot_dimmed_scenes(dimmed_scenes_timestamps, filename):
    """
    Plot the dimmed scenes as a bar graph to visualize which scenes were dimmed the most.
    
    Parameters:
    dimmed_scenes_timestamps (list of tuples): List containing tuples of (start time, stop time, dim factor)
    """
    import matplotlib.pyplot as plt
    # Extract start and end times for plotting
    start_times = [start for start, _, _ in dimmed_scenes_timestamps]
    end_times = [end for _, end, _ in dimmed_scenes_timestamps]
    dim_factors = [float(factor) for _, _, factor in dimmed_scenes_timestamps]

    # Convert start and end times to seconds for calculation
    start_times_seconds = [int(min_sec.split(':')[0]) * 60 + float(min_sec.split(':')[1]) for min_sec in start_times]
    end_times_seconds = [int(min_sec.split(':')[0]) * 60 + float(min_sec.split(':')[1]) for min_sec in end_times]

    # Calculate the duration of each dimmed scene for plotting
    durations = [end - start for start, end in zip(start_times_seconds, end_times_seconds)]
    
    plt.figure(figsize=(10, 6))
    # Plot each dimmed scene as a bar with its height representing the dim factor
    for start, duration, factor in zip(start_times_seconds, durations, dim_factors):
        plt.bar(x=start, height=factor, width=duration, align='edge', alpha=0.7)
    
    # Increase the frequency of x-axis labels by a factor of 3
    original_ticks = plt.xticks()[0]
    new_ticks = np.arange(min(original_ticks), max(original_ticks) + 1, (max(original_ticks) - min(original_ticks)) / (len(original_ticks) * 3 - 1))
    new_labels = [f"{int(s//60)}:{int(s%60):02d}" for s in new_ticks]
    plt.xticks(ticks=new_ticks, labels=new_labels, rotation=45, ha="right")
    plt.xlabel('Time (minutes:seconds)')
    plt.ylabel('Dim Factor')
    plt.title('Dimmed Scenes Visualization')
    plt.savefig(filename)
    plt.close()

def copy_subtitles(input_video_file, output_video_file):
    """
    Adds subtitle tracks from an input MKV file to another MKV video file without altering the original video and audio tracks.
    This function creates a new output file that combines the original video and audio streams with the subtitle streams from the input subtitle file.
    
    Parameters:
    input_video_file (str): Path to the original video file whose video and audio streams will remain unchanged.
    subtitle_file (str): Path to the MKV file from which subtitles will be copied.
    output_file (str): Path to the new output file that will contain the combined streams.
    """
    subtitle_file = output_video_file.rsplit('.', 1)[0] + '_subtitled.' + output_video_file.rsplit('.', 1)[1]
    try:
        # Command to combine original video/audio with the subtitles into a new file
        command = [
            'ffmpeg',
            '-y',
            '-i', output_video_file,  # Original video file
            '-i', input_video_file,  # Subtitle file
            '-map', '0:v',  # Map video stream from the first input file
            '-map', '0:a',  # Map audio streams from the first input file
            '-map', '1:s',  # Map subtitle streams from the second input file
            '-c', 'copy',  # Copy all selected streams without re-encoding
            '-c:s', 'copy',  # Ensure subtitles are copied without re-encoding
            subtitle_file  # New output file
        ]
        
        # Execute the command
        subprocess.run(command, check=True)
        
        # Overwrite the original output video file with the new file containing subtitles
        # This is because ffmpeg doesn't allow overwriting the input files
        shutil.move(subtitle_file, output_video_file)
        print("Subtitles added successfully, video and audio preserved.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to add subtitles: {e}")

# This is an optional Modal decorator
@stub.local_entrypoint()
def process_input(args):
    if(args.only_plot):
        get_dimmed_scenes(args.input_file, args.only_plot, 0)
    else:
        if args.custom_scene:
            start, end, factor = args.custom_scene
            start, end = time_to_frame(start), time_to_frame(end)
            factor = float(factor) if float(factor) > 0.0 else get_dim_factor(args.input_file, start, end)
            dimmed_scenes = [(start, end, factor)]
        else:
            dimmed_scenes = get_all_dimmed_scenes(args.input_file, args.only_plot)
        
        # Convert frame numbers to timestamps just for printing
        dimmed_scenes_timestamps = [(frame_to_time(start), frame_to_time(end), "{:.2f}".format(factor)) for start, end, factor in dimmed_scenes]
        print("Dimmed scenes (start time, stop time, dim factor): ", dimmed_scenes_timestamps)
        
        # Plot the dimmed scenes
        plot_filename = args.output_file.replace('.mkv', '_dimmed_scenes_plot.png')
        plot_dimmed_scenes(dimmed_scenes_timestamps, plot_filename)
        # process_video(args.input_file, args.output_file, dimmed_scenes)
        copy_subtitles(args.input_file, args.output_file)
            
if __name__ == "__main__":
    main()