# Anime Undimmer

This python script exactly removes the dim filters for a segment of an anime clip. For an n minute episode, it takes approximately 3n minutes to run this script.

## Installation

I use Python 3.11.4 with pip 23.1.2. It will probably run on any Python3.
```
pip install -r requirements.txt
```

## Usage

This will fully calculate then undim the calculated ranges that are determined to be dimmed and not just dark, and print the final ranges to console along with their values, and overwrite output_episode_undimmed.mkv with the undimmed ep15:

```
python color_multiply.py ep15_original.mkv output_episode_undimmed.mkv
```

This will generate a bash output similar to `jjk_s2_e15.log`.

## Tuning

Note that this script is not perfect. We are still tuning the cutoffs, so it is especially liable to un-dim merely dark scenes. It's is less likely to miss dimmed scenes, although still possible.

To show the brightness histogram over the whole video, run this relatively fast optional sanity check:
```
python color_multiply.py ep15_original.mkv output_episode_undimmed.mkv --only_plot
```

This will only display the ranges that are likely to be dark or dimmed (the dips in the graph), and plots the max brightness per frame. It will also cache those max values in a local .pkl to avoid expensive recalculation for any more computation on an input video of the same filename. It will NOT actually do any dimming computations nor write to the output file.

To dim or undim a specific scene, you can call
```
python color_multiply.py input.mp4 output.mp4 --custom_scene 0:30 1:20 1.5
```
This will defines a custom dimmed scene from 30 seconds to 1 minute 20 seconds, with a dim factor of 1.5 (meaning everything in this range will get brighter by about 50%, corresponding to the average about a 33% initial reduction for dimmed scenes). You can also add dim to scenes via multiplying by numbers less than 1 i.e. `0.7` instead of `1.5` will dim the scene by 30% instead.

This can help fine tune the result.

## How it works

What is happening when a scene is dimmed, is that a semi-opaque black filter is applied to that scene. This linearly decreases all RGB values (i.e. multiplies all by 0.8). This varies based on the epileptic value of the scene. This is why normal filters like brightness, gamma, contrast, and saturation don't work here -- none of them undo this scaling by scaling it up by an exact multiple, they all have different functions (understand more [here](https://chat.openai.com/share/e0198c0f-e20a-4eb1-a51f-cfbf085d6533)).

So we write Python to multiply each scene by a constant, that's easy and just about 5 lines.

The hard part is detecting which frames are dimmed or not.

There are a couple key insights here:
- **Detect dark frame ranges at least 15 frames long**: The maximum RGB value in an epileptic range is m. Then scaling m to 256 will scale the whole scene accordingly to normal luminescence, since we assume dimmed scenes originally have a bright white flash somewhere. If m is under some threshold, its either a dark or dimmed scene. Empirically, we noticed dimmed frames are usually dimmed at least 10%, most are dimmed about 30%, and some are dimmed as much as 50%. To handle scenes with mixes of dimmings, first we go through and calculate the 50% dims, then the 30% dims, then the 10% dims, and we update each frame in that order.
- **Detect rapidly changing frames**: To differentiate merely dark scenes and dimmed scenes, we can check how "epileptic" it is. If the frames are rapidly changing, then it's probably epileptic. After analyzing one episode, we arbitrarily set this threshold for the mean at 75 (meaning on average, each pixel changes by 75 RGB points either way for each frame diff in the sequence) and the standard deviation over 7 (all the epileptic frames we saw had std.dev over 10, and mostly over 40, but it seems fine to be conservative here). We call these two values "epileptic risk" respectively.
- **Speed via caching and generators**: To make it faster, we cache the calculated max values to be able to restart a closed process faster (in the .pkl). We also store generators instead of arrays for the expensive step of calculating mean and variance of each dimmed/dark scenes to figure out if they are epileptic, for it to be as efficient a computation as possible. It's still slow though, since it's Python.

## To-do
- Replace the cutoffs for mean and std. dev. by a simple linear regression (i.e. the higher the mean, the lower the std.dev has can be to qualify as a dimmed scene). You can use `jjk_s2_e15.log` to do this, as we think the scenes identified as dimmed are correct. (high priority, as this is critical and finnicky)
- Sometimes, dim scenes have a window in the background and briefly get undimmed mid-scene. To avoid this, if a short (say < 3 second) undimmed section between two dimmed scenes has a similar composition color-wise to the surrounding scenes, then dim it as well. This can be done via average + epileptic value being close in range (high-medium priority, as this ruins the watching experience).
- Use cython to speed it up (medium priority, as running speed is definitely a bottleneck)
- If a detected dimmed frame is a sub-part of a broader frame with the same palette (i.e. 16:55.17 - 16:55.79 in JJK S02E15), then ignore it. Low priority (it probably won't work) and hard.
- Instead of putting 256 as the numerator by which to calculate the dimming factor for each dim range, put the neighboring scene maxes. Low priority (it usually will be 256).

We welcome any contributions, comments, or collaborations!
