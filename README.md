# Anime Undimmer

This python script exactly removes the dim filters for a segment of an anime clip.

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python color_multiply.py ep15.mkv tmp.mkv --only_plot
```

This will only calculate the ranges that are likely to be dimmed, and plots the max brightness per frame.

This will fully undim the calculated ranges, and print the final ranges to console:
```
python color_multiply.py ep15.mkv tmp.mkv
```

## How it works

What is happening when a scene is dimmed, is that a semi-opaque black filter is applied to that scene. This linearly decreases all RGB values (i.e. multiplies all by 0.8). This varies based on the epileptic value of the scene. This is why normal filters like brightness, gamma, contrast, and saturation don't work here -- none of them undo this scaling by scaling it up by an exact multiple, they all have different functions.

So we write Python to multiply each scene by a constant, that's easy. The hard part is detecting which frames are dimmed or not.

There are a couple key insights and assumptions here:
- The maximum RGB value in an epileptic range is m. Then scaling m to 256 will scale the whole scene accordingly to normal luminescence, since we assume dimmed scenes originally have a bright white flash somewhere. If m is under some threshold, its either a dark or dimmed scene. Empirically, we noticed dimmed frames are usually dimmed at least 25%, so we set the threshold to 193 (out ot 256).
- To differentiate merely dark scenes and dimmed scenes, we can check how "epileptic" it is. If the frames are rapidly changing, then it's probably epileptic. After analyzing one episode, we arbitrarily set this threshold for the mean at 75 (meaning on average, each pixel changes by 75 RGB points either way for each frame diff in the sequence) and the standard deviation over 7 (all the epileptic frames I saw had std.dev over 10, and mostly over 40, but it seems fine to be conservative here).

To make it faster, we cache the calculated max values to be able to restart a closed process faster. We also store generators for the expensive step of calculating mean and variance of scenes to figure out if they are epileptic, for it to be as efficient a computation as possible.

Todo:
- If a detected dimmed frame is a sub-part of a broader frame with the same palette (i.e. 16:55.17 - 16:55.79 in JJK S02E15), then ignore it.