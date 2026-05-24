# MOG People Tracking

Foreground segmentation and lightweight people tracking using a custom mixture-of-Gaussians background model.

## Highlights

- Maintains a per-pixel Gaussian-mixture background model.
- Produces foreground masks from short image sequences.
- Extracts detections from masks and links them into simple IoU-based tracks.
- Generates debug frames, masks, a short result video, and a people count summary.

## Repository Layout

- `background_mog.py` - custom MOG background subtraction.
- `people_tracker.py` - foreground cleanup, detection extraction, tracking, and visualization.
- `imgs/` - input frame sequence.
- `examples/` - generated debug frames, masks, video, and people-count output.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python background_mog.py
python people_tracker.py --imgs_dir imgs --out_dir output_task2
```

## Tracking output

![mog-people-tracking result screenshot](docs/results/result-screenshot.png)

Foreground mask and debug tracking frame from the example image sequence.


## Background-modeling workflow

- Mixture-of-Gaussians foreground segmentation implemented for a small video-like sequence.
- Lightweight connected-component tracking and people-count output.
- Saved debug frames that make the segmentation/tracking stages inspectable.


## Follow-up validation

- The tracker is intentionally lightweight and not robust to heavy occlusion.
- The sample sequence is short and controlled.
- Next steps: add MOT-style metrics and compare against OpenCV background subtraction.

