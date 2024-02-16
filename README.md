# Anime Undimmer

This python script exactly removes the dim filters for a segment of an anime clip. For an n minute episode, it takes approximately n minutes to run this script (and about n/4 minutes for every subsequent run of the same video file).

## Installation

I use Python 3.11.4 with pip 23.1.2. It will probably run on any Python3.
```bash
pip install -r requirements.txt
python3 -m modal setup
```

## Usage

Note that the script right now only works on .mkv files, not .mp4 files (due to a mysterious speedup).

This will fully calculate then undim the calculated ranges that are determined to be dimmed and not just dark, and print the final ranges to console along with their values, and overwrite ep_undimmed.mkv with the undimmed version of ep_dimmed:

```bash
python color_multiply.py ep_dimmed.mkv --out ep_undimmed.mkv
```

This will generate a bash output similar to `jjk_s2_e15.log`.

## Tuning

Note that this script is not perfect. We are still tuning the cutoffs, so it is especially liable to un-dim merely dark scenes. It's is less likely to miss dimmed scenes, although still possible.

To show the brightness histogram over the whole video, run this relatively fast optional sanity check:
```bash
python color_multiply.py ep15_original.mkv --only_plot
```

This will display the ranges that are likely to be dark or dimmed (the dips in the graph), and plots the max luminescence per frame. It will also cache those max values in a local .pkl to avoid expensive recalculation for any more computation on an input video of the same filename. only_plot means the script will NOT actually do any dimming computations nor write to the output file.

To dim or undim a specific scene, you can call
```bash
python color_multiply.py input.mp4 --out output.mp4 --custom_scene 0:30 1:20
```
The above will define a custom dimmed scene from 30 seconds to 1 minute 20 seconds, and automatically undim it (and output the factor used).

If you think a scene should be dimmed or undimmed by a specific factor, you can call
```bash
python color_multiply.py input.mp4 --out output.mp4 --custom_scene 0:30 1:20 1.5
``` 
with a dim factor of 1.5 (meaning everything in this range will get brighter by about 50%, corresponding to the average about a 33% initial reduction for dimmed scenes). You can also add dim to scenes via multiplying by numbers less than 1 i.e. `0.7` instead of `1.5` will dim the scene by 30% instead.

This can help fine tune the result, especially if the automatic script messes up!

## How it works

What is happening when a scene is dimmed, is that a semi-opaque black filter is applied to that scene. This linearly decreases all RGB values (i.e. multiplies all by 0.8). This varies based on the epileptic value of the scene. This is why normal filters like brightness, gamma, contrast, and saturation don't work here -- none of them undo this scaling by scaling it up by an exact multiple, they all have different functions (understand more [here](https://chat.openai.com/share/e0198c0f-e20a-4eb1-a51f-cfbf085d6533)).

So we write Python to multiply each scene by a constant, that's easy and just about 5 lines.

The hard part is detecting which frames are dimmed or not.

There are a couple key insights here:
- **Detect dark frame ranges at least 15 frames long**: The maximum RGB value in an epileptic range is m. Then scaling m to 256 will scale the whole scene accordingly to normal luminescence, since we assume dimmed scenes originally have a bright white flash somewhere. If m is under some threshold, its either a dark or dimmed scene. Empirically, we noticed dimmed frames are usually dimmed at least 10%, most are dimmed about 30%, and some are dimmed as much as 50%. To handle scenes with mixes of dimmings, first we go through and calculate the 50% dims, then the 30% dims, then the 10% dims, and we update each frame in that order.
- **Detect rapidly changing frames**: To differentiate merely dark scenes and dimmed scenes, we can check how "epileptic" it is. If the frames are rapidly changing, then it's probably epileptic. We ran a simple linear regression on the values from JJK S2E15 and got thresholds for the mean at 75 (meaning on average, each pixel changes by 75 RGB points either way for each frame diff in the sequence) and the standard deviation over 7 (all the epileptic frames we saw had std.dev over 10, and mostly over 40, but it seems fine to be conservative here). We call these two values "epileptic risk" respectively.
  - A good question is, why not use the official guidelines? They say that if two frames within 8 frames change luminescence by 20, then the flashes are too fast. So this would mean that in the episode we see, after dimming by say a factor of 2x, we'd be looking to see two frames within 8 frames with a difference of 10 (now that it's been scaled down). The key problem here is that many normal dark scenes might have such small luminescence differences on their own, so following the official guidelines is basically impossible to detect.
  - But wait you might ask, what if you check that after the dimming, the flashing is gone? The answer is, it's usually actually not. There's still wild luminescence differences that, had the episode been re-processed, would have been dimmed even more. I don't know why this is passable to be honest.
- **Speed via caching and generators**: To make it faster, we cache the calculated max values to be able to restart a closed process faster (in the .pkl). We also store generators instead of arrays for the expensive step of calculating mean and variance of each dimmed/dark scenes to figure out if they are epileptic, for it to be as efficient a computation as possible. It's still slow though, since it's Python.

## To-do
- If there's two consecutive dimmed ranges for some scene (say, less than 2 seconds apart), also test the middle range for epilepsy and dimming (high priority, as incorrectly converted dark scenes end up with flashes in the middle so this is a band-aid)
- Rerun the regression with flashes and epileptic parameters on the ground truth to get the results (medium priority, as this will likely fully solve the dark vs dim problem. normally would be high but rn it's sorta mostly solved tho)
- If there's a massive dimmed range, it's possible that some subset of that is really epileptic and some subset is not. Use flashes to only undim scenes at least 15 frames long, with at least two flashes maybe? (medium priority, it seems higher priority is fix the above?)
- Sometimes, dim scenes have a window in the background and briefly get undimmed mid-scene. To avoid this, if a short (say < 3 second) undimmed section between two dimmed scenes has a similar composition color-wise to the surrounding scenes, then dim it as well. This can be done via average + epileptic value being close in range (high-medium priority, as this is a common pattern if like, a bright window appears briefly in a scene).
- Improve dim vs dark scene differentiation. One idea is to replace the cutoffs for mean and std. dev. by better linear regressions + decision trees (i.e. the higher the mean, the lower the std.dev has can be to qualify as a dimmed scene). You can use `jjk_s2_e15.log` to get those values (currently I only take into account epileptic risk). (Medium priority, as algorithmic changes might be better)
- Use cython to speed it up (medium priority, as running speed is definitely a bottleneck)
- Address the TODOs (medium-low priority, as I skipped these tasks the first time around for speed of implementation)
- If a detected dimmed frame is a sub-part of a generally dark broader scene (i.e. maybe similar mean luminances, or with the same palette), like 16:55.17 - 16:55.79 in JJK S02E15, then treat it as dark instead. (low priority and on the harder side, as flashing seems to be a better criteria to optimize)
- Instead of putting 256 as the numerator by which to calculate the dimming factor for each dim range, put the neighboring scene maxes. Low priority (it usually will be 256) but easier.

We welcome any contributions, comments, or collaborations! Especially additions to `notes.txt` with scenes that were poorly dimmer or undimmed by this script. For badly dimmed scenes, I'd appreciate uploads with the full output of the script, so I can debug the values that were used, as well as a link to the episode if possible!

## Resources

To understand seizure dimming in anime, try looking at:

These repos:
- https://github.com/Falach/epileptic_activity
- https://github.com/user234683/seizure-flash-blocker
- https://github.com/w3c/wcag/issues/553

These guidelines:
- https://www.w3.org/TR/UNDERSTANDING-WCAG20/seizure.html
- https://www.w3.org/WAI/WCAG22/Understanding/three-flashes
- ANIMATED PROGRAM IMAGE EFFECT PRODUCTION GUIDELINES: https://www.tv-tokyo.co.jp/kouhou/guideenglish.htm
- Guidance for the reduction of photosensitive epileptic seizures caused by television: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1702-0-200502-I!!PDF-E.pdf
- Harding test: https://en.wikipedia.org/wiki/Harding_test (The only implementation online is this closed source, paid program which is probably trivial to rewrite: https://hardingtest.com/)
- Photosensitive Epilepsy Analysis Tool (PEAT): https://trace.umd.edu/peat/

These tutorials:
- https://www.useragentman.com/blog/2020/07/19/how-to-fix-seizure-inducing-sequences-in-videos/

These papers:
- Automatic detection of flashing content: https://sci-hub.se/https://ieeexplore.ieee.org/document/7148104
- Japanese research trends toward biomedical assessment of digital contents: https://sci-hub.se/https://www.worldscientific.com/doi/epdf/10.1142/9781860948800_0094
- FLIKCER - Chrome Extension to Resolve Online Epileptogenic Visual Content with Real-Time Luminance Frequency Analysis: https://arxiv.org/pdf/2108.09491.pdf
- Photic- and Pattern-induced Seizures: Expert Consensus of the Epilepsy Foundation of America Working Group: https://www.epilepsy.com/sites/default/files/atoms/files/Epilepsia%20vol%2046%20issue%209%20Photosensitivity.pdf